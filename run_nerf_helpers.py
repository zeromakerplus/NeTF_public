import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from load_generated_data import load_generated_gt
from scipy.stats import multivariate_normal
from MLP import Network
import scipy.io
from scipy import linalg
import multiprocessing
import numpy.matlib


torch.autograd.set_detect_anomaly(True)
# TODO: remove this dependency
# from torchsearchsorted import searchsorted


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # # Get pdf
    # weights = weights + 1e-5 # prevent nans
    # pdf = weights / torch.sum(weights, -1, keepdim=True)
    # cdf = torch.cumsum(pdf, -1)
    # cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # # Take uniform samples
    # if det:
    #     u = torch.linspace(0., 1., steps=N_samples)
    #     u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    # else:
    #     u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # # Pytest, overwrite u with numpy's fixed random numbers
    # if pytest:
    #     np.random.seed(0)
    #     new_shape = list(cdf.shape[:-1]) + [N_samples]
    #     if det:
    #         u = np.linspace(0., 1., N_samples)
    #         u = np.broadcast_to(u, new_shape)
    #     else:
    #         u = np.random.rand(*new_shape)
    #     u = torch.Tensor(u)

    # # Invert CDF
    # u = u.contiguous()
    # inds = searchsorted(cdf, u, side='right')
    # below = torch.max(torch.zeros_like(inds-1), inds-1)
    # above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    # inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # denom = (cdf_g[...,1]-cdf_g[...,0])
    # denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    # t = (u-cdf_g[...,0])/denom
    # samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

# Spherical Sampling

def spherical_sample(t, x0, y0, z0, test_accurate_sampling, volume_position, volume_size):
    # [x0,y0,z0] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]
    v_light = 3 * 1e8
    r = v_light * t * 1e-12

    if test_accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)
        theta_min = np.min(sphere_box_point[:,1])
        theta_max = np.max(sphere_box_point[:,1])
        phi_min = np.min(sphere_box_point[:,2])
        phi_max = np.max(sphere_box_point[:,2])
        theta = np.linspace(theta_min, theta_max , 16)
        phi = np.linspace(phi_min, phi_max, 16)
        # 预设采样范围
    else:
        theta = np.linspace(0, np.pi , 16)
        phi = np.linspace(-np.pi, 0, 16)
    # 直角坐标图像参考 Zaragoza 数据集中的坐标系
    # 球坐标图像参考 Wikipedia 球坐标系词条 ISO 约定
    # theta 是 俯仰角，与 Z 轴正向的夹角， 范围从 [0,pi]
    # phi 是 在 XOY 平面中与 X 轴正向的夹角， 范围从 [-pi,pi],本场景中只用到 [-pi,0]

    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    # print(x.shape)
    x = x.flatten().reshape([-1,1])
    y = y.flatten().reshape([-1,1])
    z = z.flatten().reshape([-1,1])

    samples = np.concatenate((x*r + x0, y*r + y0, z*r + z0), axis=1)

    return samples # 注意：如果sampling正确的话，x 和 z 应当关于 x0,z0 对称， y 应当只有负值

def spherical_sample_histgram(I, L, camera_grid_positions, num_sampling_points, test_accurate_sampling, volume_position, volume_size, c, deltaT, no_rho, start, end):
    # t = np.linspace(I,L,L - I + 1) * deltaT # 
    # 输入: camera_grid_points, I
    # 输出：(L - I + 1) x 256 x 3
    [x0,y0,z0] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]
    # v_light = c #3 * 1e8
    # r = v_light * t 

    if test_accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)
        # r_min = np.min(sphere_box_point[:,0])
        # r_min = 0.1 # zaragoza 256
        # r_min = 0.35 # generated data
        # r_min = min(r_min, abs(volume_position[1] + volume_size), 0.15)
        
        # r_max = np.max(sphere_box_point[:,0]) + 0
        # r_max = min(r_max, (L - 1) * v_light * deltaT)

        r_min = 100 * c * deltaT
        r_max = 300 * c * deltaT
        theta_min = np.min(sphere_box_point[:,1]) - 0
        theta_max = np.max(sphere_box_point[:,1]) + 0
        phi_min = np.min(sphere_box_point[:,2]) - 0
        phi_max = np.max(sphere_box_point[:,2]) + 0
        theta = torch.linspace(theta_min, theta_max , num_sampling_points).float()
        phi = torch.linspace(phi_min, phi_max, num_sampling_points).float()
        dtheta = (theta_max - theta_min) / num_sampling_points
        dphi = (phi_max - phi_min) / num_sampling_points
        # 预设采样范围
    else:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)

        r_min = 0.1
        r_max = np.max(sphere_box_point[:,0])
        theta = torch.linspace(0, np.pi , num_sampling_points).float()
        phi = torch.linspace(-np.pi, 0, num_sampling_points).float()

        dtheta = (np.pi) / num_sampling_points
        dphi = (np.pi) / num_sampling_points
    # 直角坐标图像参考 Zaragoza 数据集中的坐标系
    # 球坐标图像参考 Wikipedia 球坐标系词条 ISO 约定
    # theta 是 俯仰角，与 Z 轴正向的夹角， 范围从 [0,pi]
    # phi 是 在 XOY 平面中与 X 轴正向的夹角， 范围从 [-pi,pi],本场景中只用到 [-pi,0]

    # r_min = 130 * c * deltaT # zaragoza64 bunny
    # r_max = 300 * c * deltaT

    # r_min = 100 * c * deltaT # zaragoza256 bunny # fk 和 
    # r_max = 300 * c * deltaT

    # r_min = 1 * c * deltaT # serapis
    # r_max = 500 * c * deltaT

    # r_min = 300 * c * deltaT # zaragoza256 T 
    # r_max = 500 * c * deltaT

    # r_min = 250 * c * deltaT # zaragoza256_2 
    # r_max = 450 * c * deltaT

    r_min = start * c * deltaT # zaragoza256_2 
    r_max = end * c * deltaT

    num_r = math.ceil((r_max - r_min) / (c * deltaT))
    r = torch.linspace(r_min, r_max , num_r).float()

    I1 = r_min / (c * deltaT)
    I2 = r_max / (c * deltaT)

    I1 = math.floor(I1)
    I2 = math.ceil(I2)
    I0 = r.shape[0]

    '''x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    # print(x.shape)
    x = x.flatten().reshape([-1,1])
    y = y.flatten().reshape([-1,1])
    z = z.flatten().reshape([-1,1])

    samples = np.concatenate((x*r + x0, y*r + y0, z*r + z0), axis=1)'''
    # a,b,c = torch.meshgrid(r, theta, phi)
    grid = torch.stack(torch.meshgrid(r, theta, phi),axis = -1)

    spherical = grid.reshape([-1,3])
    cartesian = spherical2cartesian_torch(spherical)
    # cartesian = cartesian + np.array([x0,y0,z0])
    cartesian = cartesian + torch.tensor([x0,y0,z0])
    if not no_rho:
        # cartesian = np.concatenate((cartesian, spherical[:,1:3]), axis = 1)
        cartesian = torch.cat((cartesian, spherical[:,1:3]), axis = 1)
    # cartesian_grid = cartesian.reshape(grid.shape)
    # cartesian_grid = cartesian_grid.reshape([L - I + 1, num_sampling_points ** 2, 3])
    #  
    a = 1
    # print(I1,I1+I0)
    return cartesian, I1, I2, I0, dtheta, dphi, theta_min, theta_max, phi_min, phi_max  # 注意：如果sampling正确的话，x 和 z 应当关于 x0,z0 对称， y 应当只有负值

def elliptic_sampling_histogram(I, L, camera_grid_positions, laser_grid_positions, num_sampling_points, test_accurate_sampling, volume_position, volume_size, c, deltaT, no_rho, start, end, device):
    cartesian = torch.zeros((end - start) * num_sampling_points ** 2, 5)
    cartesian = torch.zeros((end - start) , num_sampling_points ** 2, 5)

    S = torch.from_numpy(camera_grid_positions).float().to(device)
    L = torch.from_numpy(laser_grid_positions).float().to(device)
    center = (S + L) / 2
    # [xs,ys,zs] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]
    # [xl,yl,zl] = [laser_grid_positions[0],laser_grid_positions[1],laser_grid_positions[2]]
    # [x0,y0,z0] = [(xs + xl) / 2, (ys + yl) / 2, (zs + zl) / 2]

    delta_x = (L[0] - S[0])
    delta_z = (L[2] - S[2])
    if torch.abs(delta_x) < 0.00001:
        if torch.abs(delta_z) < 0.00001:
            beta = torch.tensor(0.00)
        elif delta_z > 0:
            beta = torch.tensor(np.pi / 2)
        elif delta_z < 0:
            beta = torch.tensor(-np.pi / 2)
    else:
        if delta_x >= 0:
            beta = torch.atan(delta_z / delta_x)
        elif (delta_x < 0) & (delta_z >= 0):
            beta = np.pi + torch.atan(delta_z / delta_x)
        elif (delta_x < 0) & (delta_z < 0):
            beta = -np.pi + torch.atan(delta_z / delta_x)
    rotation_matrix = torch.tensor([[torch.cos(beta),-torch.sin(beta)],[torch.sin(beta),torch.cos(beta)]])
    rotation_matrix_inv = torch.tensor([[torch.cos(beta),torch.sin(beta)],[-torch.sin(beta),torch.cos(beta)]])

    if test_accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point = torch.from_numpy(box_point).float().to(device)
        box_point = box_point - center
        XZ = box_point[:, [0,2]]
        XZ = (rotation_matrix_inv @ XZ.T).T
        box_point = torch.stack((XZ[:,0], box_point[:,1], XZ[:,1]), axis = 1)

        theta_min = np.pi / 2
        theta_max = np.pi / 2
        phi_min = -np.pi / 2
        phi_max = -np.pi / 2
        for i in range(box_point.shape[0]):
            box_point_elliptic = cartesian2elliptic(box_point[i,:], S, L)
            if box_point_elliptic == None:
                pass
            else:
                if box_point_elliptic[1] < theta_min:
                    theta_min = box_point_elliptic[1]
                if box_point_elliptic[1] > theta_max:
                    theta_max = box_point_elliptic[1]
                if box_point_elliptic[2] < phi_min:
                    phi_min = box_point_elliptic[2]
                if box_point_elliptic[2] > phi_max:
                    phi_max = box_point_elliptic[2]
    else: 
        theta_min = 0
        theta_max = np.pi
        phi_min = -np.pi 
        phi_max = 0
    # box_point[:,0] = box_point[:,0]
    # box_point[:,1] = box_point[:,1]
    # box_point[:,2] = box_point[:,2] 
    # xmax = np.max(box_point[:,0])
    # xmin = np.min(box_point[:,0])
    # zmax = np.max(box_point[:,2])
    # zmin = np.min(box_point[:,2])

    camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)
    laser_grid_positions = torch.from_numpy(laser_grid_positions).float().to(device)
    '''
    for l in range(start,end):
            i = l - start
            OL = l * c * deltaT
            a = torch.tensor(OL / 2)
            f = torch.sqrt(torch.sum((camera_grid_positions - laser_grid_positions) ** 2)) / 2
            b = torch.sqrt(a ** 2 - f ** 2)
            if torch.isnan(b):
                cartesian[i,:,:] = 0
            else:
                theta = torch.linspace(0, np.pi , num_sampling_points).float()
                phi = torch.linspace(-np.pi, 0, num_sampling_points).float()
                Length = torch.tensor(OL)
                o = torch.meshgrid(Length, theta, phi)
                grid = torch.stack(torch.meshgrid(Length, theta, phi),axis = -1)

                elliptic = grid[0,:,:,:].reshape([-1,3])
                grid = elliptic2cartesian_torch(elliptic, a, b, f)

                # cartesian = elliptic_rotation_torch(cartesian)
                XZ = grid[:,[0,2]]
                XZ = (rotation_matrix @ XZ.T).T
                grid = torch.stack((XZ[:,0], grid[:,1], XZ[:,1]), axis = 1)
                grid = grid + center
                if not no_rho:
                    grid = torch.cat((grid, elliptic[:,1:3]), axis = 1)
                cartesian[i,:,:] = grid
    '''
    l = torch.linspace(start, end - 1, end - start)
    OL = l * c * deltaT
    a = OL / 2
    f = torch.sqrt(torch.sum((camera_grid_positions - laser_grid_positions) ** 2)) / 2
    b = torch.sqrt(a ** 2 - f ** 2)

    # theta = torch.linspace(0, np.pi , num_sampling_points).float()
    # phi = torch.linspace(-np.pi, 0, num_sampling_points).float()
    theta = torch.linspace(theta_min, theta_max , num_sampling_points).float()
    phi = torch.linspace(phi_min, phi_max, num_sampling_points).float()
    [Theta, ol, Phi] = torch.meshgrid(OL, theta, phi)
    grid = torch.stack((Theta, ol, Phi), axis = -1)
    elliptic = grid[:,:,:,:].reshape([-1, num_sampling_points ** 2, 3])
    nan_loc = torch.where(b != b)[0]
    if nan_loc.shape[0] != 0:
        elliptic[nan_loc[0]:nan_loc[-1] + 1,:,:] = 0
        b[nan_loc[0]:nan_loc[-1] + 1] = 0

    grid = elliptic2cartesian_torch_vec(elliptic, a, b, f)
    # grid = grid.reshape(-1, num_sampling_points ** 2, 3)

    
    XZ = grid[:,[0,2]]
    XZ = (rotation_matrix @ XZ.T).T
    grid = torch.stack((XZ[:,0], grid[:,1], XZ[:,1]), axis = 1)
    grid = grid + center
    if not no_rho:
        elliptic = elliptic.reshape(-1,3)
        grid = torch.cat((grid, elliptic[:,1:3].reshape(-1, 2)), axis = 1)
    cartesian = grid
    # cartesian[i,:,:] = grid
    # if not no_rho:
    #     cartesian = cartesian.reshape(-1,5)
    #         # print(OL)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    I1 = start
    I2 = end
    I0 = I2 - I1
    return cartesian, I1, I2, I0, dtheta, dphi


def encoding(pt, L):
    # coded_pt = torch.zeros(6 * L)
    # logseq = torch.logspace(start=0, end=L-1, steps=L, base=2)
    # xsin = torch.sin(logseq.mul(math.pi).mul(pt[0]))
    # ysin = torch.sin(logseq.mul(math.pi).mul(pt[1]))
    # zsin = torch.sin(logseq.mul(math.pi).mul(pt[2]))
    # xcos = torch.cos(logseq.mul(math.pi).mul(pt[0]))
    # ycos = torch.cos(logseq.mul(math.pi).mul(pt[1]))
    # zcos = torch.cos(logseq.mul(math.pi).mul(pt[2]))
    # coded_pt = torch.reshape(torch.cat((xsin,xcos,ysin,ycos,zsin,zcos)), (1, 6 * L))

    logseq = np.logspace(start=0, stop=L-1, num=L, base=2)
    xsin = np.sin(logseq*math.pi*pt[0])
    ysin = np.sin(logseq*math.pi*pt[1])
    zsin = np.sin(logseq*math.pi*pt[2])
    xcos = np.cos(logseq*math.pi*pt[0])
    ycos = np.cos(logseq*math.pi*pt[1])
    zcos = np.cos(logseq*math.pi*pt[2])
    coded_pt = np.reshape(np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos)), (1, 6 * L))
    # i = 1
    return coded_pt

def encoding_sph(hist, L):
    # coded_hist = torch.cat([encoding(hist[k], L) for k in range(hist.shape[0])], 0)

    coded_hist = np.concatenate([encoding(hist[k], L) for k in range(hist.shape[0])], axis=0)

    return coded_hist

def encoding_batch(pt, L):
    # 输入 pt 是 N x 3 的矩阵tensor
    # coded_pt = torch.zeros(6 * L)
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2)
    xsin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,0].view(-1,1))
    xcos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,0].view(-1,1))
    ysin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,1].view(-1,1))
    ycos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,1].view(-1,1))
    zsin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,2].view(-1,1))
    zcos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,2].view(-1,1))
    coded_pt = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos),axis = 1)
    return coded_pt

def encoding_batch_tensor(pt, L, no_rho):
    # 输入 pt 是 N x 3 的矩阵numpy
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2).float()
    xsin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,0].view(-1,1))
    xcos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,0].view(-1,1))
    ysin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,1].view(-1,1))
    ycos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,1].view(-1,1))
    zsin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,2].view(-1,1))
    zcos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,2].view(-1,1))
    if no_rho:
        coded_pt = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos),axis = 1)
    else:
        thetasin = torch.sin((logseq * np.pi).reshape([1,-1]) * pt[:,3].reshape([-1,1]))
        thetacos = torch.cos((logseq * np.pi).reshape([1,-1]) * pt[:,3].reshape([-1,1]))
        phisin = torch.sin((logseq * np.pi).reshape([1,-1]) * pt[:,4].reshape([-1,1]))
        phicos = torch.cos((logseq * np.pi).reshape([1,-1]) * pt[:,4].reshape([-1,1]))
        coded_pt = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos,thetasin,thetacos,phisin,phicos),axis = 1)
    return coded_pt

def encoding_batch_numpy(pt, L, no_rho):
    # 输入 pt 是 N x 3 的矩阵numpy
    logseq = np.logspace(start=0, stop=L-1, num=L, base=2)
    xsin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,0].reshape([-1,1]))
    xcos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,0].reshape([-1,1]))
    ysin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,1].reshape([-1,1]))
    ycos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,1].reshape([-1,1]))
    zsin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,2].reshape([-1,1]))
    zcos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,2].reshape([-1,1]))
    if no_rho:
        coded_pt = np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos),axis = 1)
    else:
        thetasin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,3].reshape([-1,1]))
        thetacos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,3].reshape([-1,1]))
        phisin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,4].reshape([-1,1]))
        phicos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,4].reshape([-1,1]))
        coded_pt = np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos,thetasin,thetacos,phisin,phicos),axis = 1)
    return coded_pt



def nlos_render(samples, camera_gridposition, model, use_encoding, encoding_dim,volume_position, volume_size):
    sphere_res = 0 # sphere_res 是nerf渲染出的单个bin的结果， 是一个torch.Tensor标量
    distance_square = camera_gridposition[0] ** 2 + camera_gridposition[1] ** 2 + camera_gridposition[2] ** 2
    for l in range(len(samples[0])):
        x = samples[l, 0] + volume_size[0] / (2 * volume_size[0])
        y = samples[l, 1] - volume_position[1] / (2 * volume_size[0])
        z = samples[l, 2] + volume_size[0] / (2 * volume_size[0])
        pt = torch.tensor([x, y, z], dtype=torch.float32).view(-1)
        # print(pt)
        if use_encoding:
            coded_pt = encoding(pt, L=encoding_dim) # encoding 函数将长度为 3 的 tensor 返回为长度为 6L 的tensor
            network_res = model(coded_pt)
        else:
            network_res = model(pt)
        sphere_res = sphere_res + network_res[0] / distance_square

    return sphere_res

def show_samples(samples,volume_position,volume_size, camera_grid_positions):
    # data shuffeler
    
    samples = samples[np.random.permutation(samples.shape[0])] 


    box = volume_box_point(volume_position,volume_size)
    showstep = 1

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(0,0,0,c='k',linewidths=0.03)
    ax.scatter(camera_grid_positions[0], camera_grid_positions[1], camera_grid_positions[2], c = 'g', linewidths = 0.03)
    ax.scatter(box[:,0],box[:,1],box[:,2], c = 'b', linewidths=0.03)
    ax.scatter(volume_position[0],volume_position[1],volume_position[2],c = 'b', linewidths=0.03)
    ax.scatter(samples[1:-1:showstep,0],samples[1:-1:showstep,1],samples[1:-1:showstep,2],c='r',alpha = 0.2, linewidths=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    plt.savefig('./scatter_samples')
    plt.close()

    plt.scatter(0,0,c='k',linewidths=0.03)
    plt.scatter(camera_grid_positions[0], camera_grid_positions[1], c = 'g', linewidths = 0.03)
    plt.scatter(box[:,0],box[:,1], c = 'b', linewidths=0.03)
    plt.scatter(volume_position[0],volume_position[1],c = 'b', linewidths=0.03)
    plt.scatter(samples[1:-1:showstep,0],samples[1:-1:showstep,1],c='r',alpha = 0.2, linewidths=0.01)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    plt.savefig('./scatter_samples_XOY')
    plt.close()

    plt.scatter(0,0,c='k',linewidths=0.03)
    plt.scatter(camera_grid_positions[0], camera_grid_positions[2], c = 'g', linewidths = 0.03)
    plt.scatter(box[:,0],box[:,2], c = 'b', linewidths=0.03)
    plt.scatter(volume_position[0],volume_position[2],c = 'b', linewidths=0.03)
    plt.scatter(samples[1:-1:showstep,0],samples[1:-1:showstep,2],c='r',alpha = 0.2, linewidths=0.01)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.show()
    plt.savefig('./scatter_samples_XOZ')
    plt.close()

    return 0

def volume_box_point(volume_position, volume_size):
    # 格式: volume_position: 3, 向量  volume_size: 标量
    # 输出: box: 8 x 3
    [xv, yv, zv] = [volume_position[0], volume_position[1], volume_position[2]]
    # xv, yv, zv 是物体 volume 的中心坐标
    x = np.array([xv - volume_size / 2, xv - volume_size / 2, xv - volume_size / 2, xv - volume_size / 2, xv + volume_size / 2, xv + volume_size / 2, xv + volume_size / 2, xv + volume_size / 2])
    y = np.array([yv - volume_size / 2, yv - volume_size / 2, yv + volume_size / 2, yv + volume_size / 2, yv - volume_size / 2, yv - volume_size / 2, yv + volume_size / 2, yv + volume_size / 2])
    z = np.array([zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2])
    box = np.stack((x, y, z),axis = 1)
    return box

def cartesian2spherical(pt):
    # 函数将直角坐标系下的点转换为球坐标系下的点
    # 输入格式： pt 是一个 N x 3 的 ndarray
    # print('time of C2S')
    # time1 = time.time()
    spherical_pt = np.zeros(pt.shape)
    spherical_pt[:,0] = np.sqrt(np.sum(pt ** 2,axis=1))
    # print(time.time() - time1)
    spherical_pt[:,1] = np.arccos(pt[:,2] / spherical_pt[:,0])
    # print(time.time() - time1)
    phi_yplus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] >= 0)
    phi_yplus = phi_yplus + (phi_yplus < 0).astype(np.int) * (np.pi)
    phi_yminus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] < 0)
    phi_yminus = phi_yminus + (phi_yminus > 0).astype(np.int) * (-np.pi)
    spherical_pt[:,2] = phi_yminus + phi_yplus
    # print(time.time() - time1)

    # spherical_pt[:,2] = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) 
    # spherical_pt[:,2] = spherical_pt[:,2] + (spherical_pt[:,2] > 0).astype(np.int) * (-np.pi)

    return spherical_pt

def cartesian2elliptic(pt, S, L):
    # pt: (3,)
    # S: (3,)
    # L: (3,)
    f = torch.sqrt(torch.sum((L - S) ** 2)) / 2

    L = torch.sqrt(torch.sum((pt - S) ** 2)) + torch.sqrt(torch.sum((pt - L) ** 2))
    a = L / 2
    b = torch.sqrt(a ** 2 - f ** 2)
    if torch.isnan(b):
        return None
    else:
        elliptic_pt = torch.zeros(3)
        elliptic_pt[0] = L
        # print(time.time() - time1)
        elliptic_pt[1] = torch.acos(pt[2] / b)
        # print(time.time() - time1)
        phi_yplus = (torch.atan(pt[1] / (pt[0] + 1e-8))) * (pt[1] >= 0)
        phi_yplus = phi_yplus + (phi_yplus < 0).int() * (np.pi)
        phi_yminus = (torch.atan(pt[1] / (pt[0] + 1e-8))) * (pt[1] < 0)
        phi_yminus = phi_yminus + (phi_yminus > 0).int() * (-np.pi)
        elliptic_pt[2] = phi_yminus + phi_yplus
        # print(time.time() - time1)

        # spherical_pt[:,2] = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) 
        # spherical_pt[:,2] = spherical_pt[:,2] + (spherical_pt[:,2] > 0).astype(np.int) * (-np.pi)

    return elliptic_pt

def spherical2cartesian(pt):
    # 函数将球坐标系下的点转换为直角坐标系下的点
    # 输入格式： pt 是一个 N x 3 的 ndarray
    # 测试用
    # pt = np.array([[1,0,0],[1,np.pi/2,0],[1,0,np.pi/2]])


    cartesian_pt = np.zeros(pt.shape)
    cartesian_pt[:,0] = pt[:,0] * np.sin(pt[:,1]) * np.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0] * np.sin(pt[:,1]) * np.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0] * np.cos(pt[:,1])

    return cartesian_pt

def spherical2cartesian_torch(pt):
    # 函数将球坐标系下的点转换为直角坐标系下的点
    # 输入格式： pt 是一个 N x 3 的 tensor
    # 测试用
    # pt = np.array([[1,0,0],[1,np.pi/2,0],[1,0,np.pi/2]])


    cartesian_pt = torch.zeros(pt.shape)
    cartesian_pt[:,0] = pt[:,0] * torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0] * torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0] * torch.cos(pt[:,1])

    return cartesian_pt
def elliptic2cartesian_torch(pt, a, b, f):
    # pt: N x 3, tensor
    N = pt.shape[0]
    A = a.repeat(N)
    B = b.repeat(N)
    F = f.repeat(N)

    cartesian_pt = torch.zeros(pt.shape)
    cartesian_pt[:,0] = A * torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    cartesian_pt[:,1] = B * torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    cartesian_pt[:,2] = F * torch.cos(pt[:,1])
    return cartesian_pt

def elliptic2cartesian_torch_vec(pt, a, b, f):
    # pt: L x N^2 x 3, tensor
    # a: L,
    # b: L,
    a = a.reshape(-1, 1, 1)
    b = b.reshape(-1, 1, 1)
    f = f.reshape(1, 1, 1)

    L = pt.shape[0]
    N2 = pt.shape[1]

    A = a.repeat(1, N2, 1)
    B = b.repeat(1, N2, 1)
    F = f.repeat(L, N2, 1)
    
    pt_extra = torch.cat((A,B,F,pt), axis = 2)
    pt_extra = pt_extra.reshape(-1, 6)
    cartesian_pt = torch.zeros(pt_extra.shape[0], 3)
    cartesian_pt[:,0] = pt_extra[:,0] * torch.sin(pt_extra[:,4]) * torch.cos(pt_extra[:,5])
    cartesian_pt[:,1] = pt_extra[:,1] * torch.sin(pt_extra[:,4]) * torch.sin(pt_extra[:,5])
    cartesian_pt[:,2] = pt_extra[:,1] * torch.cos(pt_extra[:,4])
    # cartesian_pt = torch.zeros(pt.shape)
    # cartesian_pt[:,0] = A * torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    # cartesian_pt[:,1] = B * torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    # cartesian_pt[:,2] = F * torch.cos(pt[:,1])
    return cartesian_pt

def threshold_bin(nlos_data):
    data_sum = torch.sum(torch.sum(nlos_data, dim = 2),dim = 1)
    for i in range(0, 800, 10):
        
        # print(i)
        # print(value)
        if (data_sum[i] < 1e-12) & (data_sum[i+10] > 1e-12):
            break 
        
    threshold_former_bin = i - 10
    if threshold_former_bin > 650:
        error('error: threshold too large')
    return threshold_former_bin

def test_set_error(model, volume, volume_vector, use_encoding , encoding_dim, batchsize):
    [xv,yv,zv] = volume_vector
    volume_location = np.meshgrid(xv,yv,zv)
    volume_location = np.stack(volume_location, axis=-1)
    volume_location = np.transpose(volume_location, (1,0,2,3))
    volume_location = volume_location.reshape([-1,3])
    volume_location = torch.from_numpy(volume_location).float()
    if use_encoding:
        volume_location = encoding_batch(volume_location,encoding_dim)
    volume = volume.reshape([-1])
    volume = torch.from_numpy(volume).float()
    N = volume.shape[0]
    error = 0
    criterion = torch.nn.L1Loss()
    for i in range(int(N / batchsize)):
        v = volume[0 + i * batchsize:batchsize + i * batchsize]
        with torch.no_grad():
            p = model(volume_location[0 + i * batchsize:batchsize + i * batchsize])
        lo = criterion(v,p)
        error = error + lo
        print(i,'/',int(N / batchsize),' loss = ',error)
    error = error / int(N / batchsize)
    return error

def compute_loss(M,m,N,n,i,j,I,L,pmin,pmax,device,criterion,model,args,nlos_data,volume_position, volume_size, c, deltaT, camera_grid_positions,s2,transform_matrix, transform_vector, *, laser_grid_positions = 0):
    [x0,y0,z0] = [camera_grid_positions[0, m * N + n + j], camera_grid_positions[1, m * N + n + j],camera_grid_positions[2, m * N + n + j]]

    if args.test_neglect_former_bins:
        with torch.no_grad():
            if args.confocal:
                [input_points, I1, I2, I0, dtheta, dphi, theta_min, theta_max, phi_min, phi_max] = spherical_sample_histgram(I, L, camera_grid_positions[:,m * N + n + j], args.num_sampling_points, args.test_accurate_sampling, volume_position, volume_size, c, deltaT, args.no_rho, args.start, args.end)
            elif not args.confocal:
                # camera_grid_positions[:,m * N + n + j] = laser_grid_positions[:,m * N + n + j]
                [input_points, I1, I2, I0, dtheta, dphi] = elliptic_sampling_histogram(I, L, camera_grid_positions[:,m * N + n + j], laser_grid_positions[:,m * N + n + j], args.num_sampling_points, args.test_accurate_sampling, volume_position, volume_size, c, deltaT, args.no_rho, args.start, args.end, device)
            
            # show samples
            # show_samples(input_points.cpu().numpy(), volume_position, volume_size, camera_grid_positions[:,m * N + n])
            # normalization
            input_points_ori = input_points[:,0:3]
            input_points = (input_points - pmin) / (pmax - pmin)

            if args.use_encoding:
                input_points_coded = encoding_batch_tensor(input_points, args.encoding_dim, args.no_rho)
                # input_points_coded = torch.from_numpy(input_points_coded).float().to(device)
        network_res = model(input_points_coded)

        if args.prior:
            density = network_res[:,1].reshape(I0, args.num_sampling_points ** 2)

        # plt.figure()
        # plt.imshow(network_res[:,1].reshape(I0, args.num_sampling_points ** 2).detach().cpu().numpy())
        # plt.show()

        if args.refinement:
        # if True:
            # print(args.refinement)
            # occlusion = torch.zeros(I0, args.num_sampling_points ** 2)
            network_res[:,1] = network_res[:,1] * (network_res[:,1] >= 0.03) +  (network_res[:,1] < 0.03) * (0.01 * network_res[:,1])
            density = network_res[:,1].reshape(I0, args.num_sampling_points ** 2)
            # density = density +  (density < 0.1) * (10 * density ** 2 - density)
            
            # for k in range(int(I0)):
            #     if k == 0:
            #         occlusion[k,:] = density[k,:]
            #     else:
            #         # occlusion[k,:] = occlusion[k-1,:] + density[k,:]
            #         occlusion[k,:] = torch.sum(density[0:(k+1),:], axis = 0)
            Down_num = args.Down_num
            occlusion = torch.zeros(int(I0 / Down_num), args.num_sampling_points ** 2)
            for k in range(int(I0 / Down_num)):
                if k <= 1:
                    occlusion[k,:] = torch.sum(density[k:(k + Down_num),:], axis = 0)
                else:
                    # occlusion[k,:] = occlusion[k-1,:] + density[k,:]
                    occlusion[k,:] = torch.sum(density[0:((k-2)*Down_num),:], axis = 0)
            occlusion = torch.repeat_interleave(occlusion, Down_num, axis = 0)
            # occlusion = torch.exp(-occlusion)
            # occlusion = (1 - occlusion) * (occlusion < 1) + 1e-4 * (1 - occlusion) * (occlusion >= 1)
            occlusion = (1 - occlusion) * (occlusion < 1) + 1e-4 * (1 - occlusion) * (occlusion >= 1)

            # plt.figure()
            # plt.imshow(density.reshape(I0, args.num_sampling_points ** 2).detach().cpu().numpy())
            # plt.show()

            # plt.figure()
            # plt.imshow(occlusion.detach().cpu().numpy())
            # plt.show()



            occlusion = occlusion.reshape(-1)
            network_res[:,1] *= occlusion

            # plt.figure()
            # plt.imshow(network_res[:,1].reshape(I0, args.num_sampling_points ** 2).detach().cpu().numpy())
            # plt.show()
        
        if (not args.reflectance) & (not args.density):
            if not args.no_rho:
                network_res = torch.prod(network_res, axis = 1)
        elif args.reflectance:
            network_res = network_res[:,0]
        elif args.density:
            network_res = network_res[:,1]

        # network_res = network_res.reshape(I0, args.num_sampling_points ** 2)
        # plt.figure()
        # plt.imshow(network_res.reshape(I0, args.num_sampling_points ** 2).detach().cpu().numpy())
        # plt.show()
        
        if args.attenuation:
            if args.confocal:
                network_res = network_res.reshape(I0, args.num_sampling_points ** 2)
                with torch.no_grad():
                    distance = (torch.linspace(I1, I2, I0) * deltaT * c).float().to(device)
                    distance = distance.reshape(-1, 1)
                    distance = distance.repeat(1, args.num_sampling_points ** 2)
                    Theta = input_points.reshape(-1, args.num_sampling_points ** 2, 5)[:,:,3]
                network_res = network_res / (distance ** 2) / torch.sin(Theta)
                network_res = network_res * (volume_position[1] ** 2) * 1
                # nlos_histogram = nlos_histogram * (distance ** 4)
            elif not args.confocal:
                # input_points_ori
                with torch.no_grad():
                    SP = torch.from_numpy(camera_grid_positions[:,m * N + n + j]).float().to(device).reshape(1,3)
                    LP = torch.from_numpy(laser_grid_positions[:,m * N + n + j]).float().to(device).reshape(1,3)
                    distance1_square = torch.sum((input_points_ori - LP) ** 2, axis = 1)
                    distance2_square = torch.sum((input_points_ori - SP) ** 2, axis = 1)
                network_res = network_res / distance1_square / distance2_square
                network_res = network_res * (volume_position[1] ** 4)
                network_res = network_res.reshape(I0, args.num_sampling_points ** 2)
        else:
            network_res = network_res.reshape(I0, args.num_sampling_points ** 2)

        # plt.figure()
        # plt.imshow(network_res.reshape(I0, args.num_sampling_points ** 2).detach().cpu().numpy())
        # plt.show()
        
        pred_histgram = torch.sum(network_res, axis = 1)

        pred_histgram = pred_histgram * dtheta * dphi

        if args.hierarchical_sampling:
            with torch.no_grad():
                # print(dtheta, dphi)
                network_res = network_res.reshape(I0, args.num_sampling_points, args.num_sampling_points)
                # time0 = time.time()
                input_points_extra, samples_pdf = MCMC_sampling(network_res, theta_min, theta_max, phi_min, phi_max, device, camera_grid_positions[:,m * N + n + j], c, deltaT, args.no_rho, args.start, args.end, args)
                # print('MCMC_sampling: ', time.time() - time0,'s')

                # show_input_points_extra = input_points_extra.reshape(I0, args.num_sampling_points ** 2, input_points_extra.shape[1]).cpu().detach().numpy()
                # for k in range(I0):
                #     print(k)
                #     plt.hist2d(show_input_points_extra[k,:,4], show_input_points_extra[k,:,3], bins = 32) # 由于显示图像的问题，要交换xy轴来显示
                #     plt.xlim(phi_min,phi_max)
                #     plt.ylim(theta_max,theta_min)#norm=norm, 
                #     # plt.colorbar(matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.Spectral_r))
                #     plt.colorbar()
                #     plt.savefig('./figure/test/samples_hist/hist_samples_' + str(k) + '.png')
                #     plt.close()
                #     plt.imshow(network_res[k,:,:].detach().cpu().numpy())
                #     plt.colorbar()
                #     plt.savefig('./figure/test/samples_pdf/pdf_' + str(k) + '.png')
                #     plt.close()
                
                # show samples
                # show_samples(input_points_extra, volume_position, volume_size, camera_grid_positions[:,m * N + n])
                # normalization

                # [input_points_extra, I1_extra, I2_extra, I0_extra, dtheta_extra, dphi_extra] = spherical_sample_histgram(I, L, camera_grid_positions[:,m * N + n + j], args.num_sampling_points, True, volume_position, volume_size, c, deltaT, args.no_rho)
                input_points_extra = (input_points_extra - pmin) / (pmax - pmin)
                if args.use_encoding:
                    input_points_coded_extra = encoding_batch_tensor(input_points_extra, args.encoding_dim, args.no_rho)
            network_res_extra = model(input_points_coded_extra)
            if not args.no_rho:
                network_res_extra = torch.prod(network_res_extra, axis = 1)
            network_res_extra = network_res_extra.reshape(I0, args.num_MCMC_sampling_points ** 2)
            samples_pdf = samples_pdf.reshape(I0, args.num_MCMC_sampling_points ** 2)

            network_res_extra = network_res_extra / samples_pdf
            '''
            if args.attenuation:
                if args.confocal:
                    with torch.no_grad():
                        distance = np.linspace(I1, I2, I0) * deltaT * c
                        distance = torch.from_numpy(distance).float().to(device)
                        nlos_histogram = nlos_histogram * (distance ** 4)
                elif not args.confocal:
                    pass
            '''
            pred_histgram_extra = torch.sum(network_res_extra, axis = 1) / (args.num_MCMC_sampling_points ** 2)
            # pred_histgram_extra = torch.sum(network_res_extra, axis = 1)
            # pred_histgram_extra = pred_histgram_extra * dtheta_extra * dphi_extra

            # plt.plot(pred_histgram.cpu().detach().numpy(), alpha = 0.5, label = 'quarture')
            # plt.plot(pred_histgram_extra.cpu().detach().numpy(), alpha = 0.5, label='MCMC')
            # plt.legend(loc='upper right')
            # plt.savefig('./figure/test/pred_histogram_quad_VS_MCMC.png')
            # plt.close()

            # plt.plot(pred_histgram.cpu().detach().numpy(), alpha = 0.5, label = 'quarture')
            # plt.legend(loc='upper right')
            # plt.savefig('./figure/test/pred_histogram_quad.png')
            # plt.close()

            # plt.plot(pred_histgram_extra.cpu().detach().numpy(), alpha = 0.5, label = 'MCMC')
            # plt.legend(loc='upper right')
            # plt.savefig('./figure/test/pred_histogram_MCMC.png')
            # plt.close()

            pred_histgram = (pred_histgram + pred_histgram_extra) / 2
        
        with torch.no_grad():
            nlos_histogram = nlos_data[I1:(I1 + I0), m, n + j]

        '''
        if args.attenuation:
            if args.confocal:
                with torch.no_grad():
                    distance = np.linspace(I1, I2, I0) * deltaT * c
                    distance = torch.from_numpy(distance).float().to(device)
                    nlos_histogram = nlos_histogram * (distance ** 4)
            elif not args.confocal:
                pass
        '''


    else:
        
        pass

    with torch.no_grad():
        nlos_histogram = nlos_histogram * 1
        # # nlos_histogram = nlos_histogram / 80 # generated sampling 16 with attenuation
        # # nlos_histogram = nlos_histogram / 400 # generated sampling 32
        # # nlos_histogram = nlos_histogram / 150 # generated sampling 64
        # nlos_histogram = nlos_histogram * 100000 # zaragoza 256 T sampling 32
        # # nlos_histogram = nlos_histogram * 1600 # zaragoza256 sampling 16
        # nlos_histogram = nlos_histogram * 10000    # zaragoza256 sampling 32
        # # nlos_histogram = nlos_histogram * 30000 # zaragoza256 sampling 64

        # nlos_histogram = nlos_histogram * 100    # zaragoza256 sampling 32 hierical samplimg
        # nlos_histogram = nlos_histogram / 100 # fk dragon dataset
        nlos_histogram = nlos_histogram * args.gt_times

    loss1 = criterion(pred_histgram, nlos_histogram)
    # loss2 = criterion(pred_histgram_extra, nlos_histogram) 
    # loss2 = 1e-1 * criterion(network_res, torch.zeros(network_res.shape).to(device))
    
    # pred_histgram_conv = pred_histgram.view(1,1,I0)
    # nlos_histogram_conv = nlos_histogram.view(1,1,I0)
    # pred_histgram_conv = F.conv1d(pred_histgram_conv,s2)
    # nlos_histogram_conv = F.conv1d(nlos_histogram_conv,s2)
    # pred_histgram_conv = pred_histgram_conv.view(-1)
    # nlos_histogram_conv = nlos_histogram_conv.view(-1)
    # loss1 = criterion(pred_histgram_conv, nlos_histogram_conv)
    # if ((n % 32 == 0) & (j == 0)) | ((m < 1) & (n < 64) & (j == 0)):
    #     plt.plot(nlos_histogram_conv.cpu(), alpha = 0.5, label = 'data')
    #     plt.plot(pred_histgram_conv.cpu().detach().numpy(), alpha = 0.5, label='predicted')
    #     # plt.plot(pred_histgram_extra.cpu().detach().numpy(), alpha = 0.5, label='predicted extra')
    #     plt.legend(loc='upper right')
    #     plt.title('grid position:' + str(x0) + ' ' + str(z0))
    #     plt.savefig('./figure/' + str(i) + '_' + str(m) + '_' + str(n) + '_' + str(j) + 'histogram_conv')
    #     plt.close()

    # pred_histgram_PCA = pred_histgram.view(1,I0)
    # nlos_histogram_PCA = nlos_histogram.view(1,I0)
    # pred_histgram_a = torch.mm((pred_histgram_PCA - transform_vector), transform_matrix)
    # nlos_histogram_a = torch.mm((nlos_histogram_PCA - transform_vector), transform_matrix)
    # pred_histgram_b = pred_histgram_a.view(-1)
    # nlos_histogram_b = nlos_histogram_a.view(-1)
    # loss1 = criterion(pred_histgram_b, nlos_histogram_b)
    # # # sample = nlos_data[:,45,75].reshape([1,-1])
    # # # transformed_sample = torch.mm((sample - transform_vector), transform_matrix)
    # # # sample_re = torch.mm(transform_matrix, transformed_sample.T).T + transform_vector
    # if ((n % 32 == 0) & (j == 0)) | ((m < 1) & (n < 64) & (j == 0)):
    #     plt.plot(nlos_histogram_b.cpu().detach().numpy(), alpha = 0.5, label = 'data')
    #     plt.plot(pred_histgram_b.cpu().detach().numpy(), alpha = 0.5, label='predicted')
    #     # plt.plot(pred_histgram_extra.cpu().detach().numpy(), alpha = 0.5, label='predicted extra')
    #     plt.legend(loc='upper right')
    #     plt.title('grid position:' + str(x0) + ' ' + str(z0))
    #     plt.savefig('./figure/' + str(i) + '_' + str(m) + '_' + str(n) + '_' + str(j) + 'histogram_PCA')
    #     plt.close()
        
        
    # sample = nlos_histogram.reshape([1,-1])
    # transformed_sample = torch.mm((sample - transform_vector), transform_matrix)
    # sample_re = torch.mm(transform_matrix, transformed_sample.T).T + transform_vector

    # plt.plot(sample.reshape([-1]).cpu().numpy(),linewidth=0.1,marker = 'x',markersize=1, label = 'sample')
    # plt.plot(sample_re.reshape([-1]).cpu().numpy(),linewidth=0.1,marker = 'o',markersize=1, label = 'sample_re')
    # plt.legend(loc='upper right')
    # plt.savefig('./figure/test/PCA1_samples' + str(n) + '.png')
    # plt.close()

    # print(torch.sum((sample - sample_re) ** 2))
    # plt.plot(transformed_sample.reshape([-1]).cpu().numpy(), label = 'feature')
    # plt.legend(loc='upper right')
    # plt.savefig('./figure/test/PCA2_samples' + str(n) + '.png')
    # plt.close()
        
    # if j == 0:
    #     loss = loss1 #+ loss2
    #     loss_batch = loss #+ loss2 #+ loss2 # 注意：loss 函数内部已经集成了平均值处理
    # else:
    #     loss = loss1 #+ loss2
    #     loss_batch += loss #+ loss2
    # print(j,loss.item())
    if args.prior:
        # a = density
        # b = torch.zeros(density.shape).to(device)
        cri2 = nn.L1Loss()
        loss2 = args.prior_para * cri2(density,torch.zeros(density.shape).to(device))
        loss = loss1 + loss2
    else:
        loss = loss1 #+ loss2
    loss_coffe = torch.mean(nlos_histogram ** 2)
    equal_loss = loss / loss_coffe

    if args.save_fig:
        if ((n % 256 == 0) & (j == 0)) | ((m < 1) & (n < 32)):
            loss_show = equal_loss.cpu().detach().numpy()
            plt.plot(nlos_histogram.cpu(), alpha = 0.5, label = 'data')
            plt.plot(pred_histgram.cpu().detach().numpy(), alpha = 0.5, label='predicted')
            # plt.plot(pred_histgram_extra.cpu().detach().numpy(), alpha = 0.5, label='predicted extra')
            plt.legend(loc='upper right')
            # plt.title('grid position:' + str(x0) + ' ' + str(z0))
            plt.title('grid position:' + str(format(x0, '.4f')) + ' ' + str(format(z0, '.4f')) + ' equal loss:' + str(format(loss_show, '.8f')) + ' coffe:' + str(format(loss_coffe.cpu().detach().numpy(), '.8f')))
            format(x0, '.3f')
            plt.savefig('./figure/' + str(i) + '_' + str(m) + '_' + str(n) + '_' + str(j) + 'histogram')
            plt.close()


    mdic = {'nlos':nlos_histogram.cpu().detach().numpy(),'pred':pred_histgram.cpu().detach().numpy()}
    scipy.io.savemat('./loss_compare.mat', mdic)


    return loss, equal_loss

def save_volume(model,coords, pmin, pmax,args, P,Q,R,device,test_batchsize,xv,yv,zv,i,m):
    pmin = pmin.cpu().numpy()
    pmax = pmax.cpu().numpy()
    # save predicted volume
    # normalization
    with torch.no_grad():
        test_input = (coords - pmin) / (pmax - pmin)

        if (not args.reflectance) & (not args.density):    
            if not args.no_rho:
                test_output = torch.empty(P * Q * R, 2).to(device)
                for l in range(int(P * Q * R / test_batchsize)):
                    test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                    test_input_batch = encoding_batch_numpy(test_input_batch, args.encoding_dim, args.no_rho)
                    test_input_batch = torch.from_numpy(test_input_batch).float().to(device)
                    test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize, :] = model(test_input_batch)
                test_volume = test_output[:,1].view(P,Q,R)
                test_volume_rho = test_output[:,0].view(P,Q,R)
                test_volume = test_volume.cpu().numpy()
                test_volume_rho = test_volume_rho.cpu().numpy()
            else:
                test_output = torch.empty(P * Q * R).to(device)
                for l in range(int(P * Q * R / test_batchsize)):
                    test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                    test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize] = model(test_input_batch).view(-1)
                test_volume = test_output.view(P,Q,R)
                test_volume = test_volume.cpu().numpy()
        elif args.reflectance:
            num_batchsize = 64 * 128
            Na = 8
            Ni = 64 * 64 * num_batchsize
            theta = np.linspace(0, np.pi, Na)
            phi = np.linspace(-np.pi, 0, Na)
            [Phi, Theta] = np.meshgrid(phi, theta)
            angle_grid = np.stack((Theta, Phi), axis = 0)
            angle_grid = np.reshape(angle_grid, [2, Na ** 2]).T
            angle_grid = np.tile(angle_grid, [num_batchsize, 1])
            test_output = torch.zeros(P * Q * R, 2).to(device)
            time0 = time.time()
            for l in range(int(P * Q * R / num_batchsize)):
                test_input_batch = test_input[0 + l * num_batchsize :num_batchsize + l * num_batchsize,:]
                test_input_batch = np.repeat(test_input_batch, Na ** 2, axis = 0)
                test_input_batch[:,3:] = angle_grid
                test_input_batch = encoding_batch_numpy(test_input_batch, args.encoding_dim, args.no_rho)
                test_input_batch = torch.from_numpy(test_input_batch).float().to(device)
                test_output_batch = model(test_input_batch)
                test_output_batch = torch.reshape(test_output_batch, [-1, Na ** 2, 2])
                test_output_batch = torch.sum(test_output_batch, axis = 1)
                test_output[0 + l * num_batchsize :num_batchsize + l * num_batchsize, 1] = test_output_batch[:,0]
                if l % int(P * Q * R / num_batchsize / 10) == 0:
                    time1 = time.time()
                    print(l)
                    print(time1 - time0)
                    time0 = time.time()
            test_output[:,1] /= (np.pi ** 2)

            for l in range(int(P * Q * R / test_batchsize)):
                test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                test_input_batch = encoding_batch_numpy(test_input_batch, args.encoding_dim, args.no_rho)
                test_input_batch = torch.from_numpy(test_input_batch).float().to(device)
                test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize, 0] = model(test_input_batch)[:, 0]
            test_volume = test_output[:,1].view(P,Q,R)
            test_volume_rho = test_output[:,0].view(P,Q,R)
            test_volume = test_volume.cpu().numpy()
            test_volume_rho = test_volume_rho.cpu().numpy()
        elif args.density:
            test_output = torch.empty(P * Q * R, 2).to(device)
            for l in range(int(P * Q * R / test_batchsize)):
                test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                test_input_batch = encoding_batch_numpy(test_input_batch, args.encoding_dim, args.no_rho)
                test_input_batch = torch.from_numpy(test_input_batch).float().to(device)
                test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize, :] = model(test_input_batch)
            test_volume = test_output[:,1].view(P,Q,R)
            test_volume_rho = test_output[:,0].view(P,Q,R)
            test_volume = test_volume.cpu().numpy()
            test_volume_rho = test_volume_rho.cpu().numpy()

    mdic = {'volume':test_volume, 'x':xv, 'y':yv, 'z':zv, 'volume_rho':test_volume_rho}
    scipy.io.savemat('./model/predicted_volume' + str(i) +'_'+ str(m) + '.mat', mdic)
    print('save predicted volume in epoch ' + str(i))

    if P == args.final_volume_size:
        XOY_density = np.max(test_volume, axis = 0)
        plt.imshow(XOY_density)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_density_XOY.png')
        plt.close()
        YOZ_density = np.max(test_volume, axis = 1)
        plt.imshow(YOZ_density)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_density_YOZ.png')
        plt.close()
        XOZ_density = np.max(test_volume, axis = 2)
        plt.imshow(XOZ_density)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_density_XOZ.png')
        plt.close()

        XOY_reflectance = np.max(test_volume_rho, axis = 0)
        plt.imshow(XOY_reflectance)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_reflectance_XOY.png')
        plt.close()
        YOZ_reflectance = np.max(test_volume_rho, axis = 1)
        plt.imshow(YOZ_reflectance)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_reflectance_YOZ.png')
        plt.close()
        XOZ_reflectance = np.max(test_volume_rho, axis = 2)
        plt.imshow(XOZ_reflectance)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_reflectance_XOZ.png')
        plt.close()
        XOY_albedo = np.max(test_volume * test_volume_rho, axis = 0)
        plt.imshow(XOY_albedo)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_albedo_XOY.png')
        plt.close()
        YOZ_albedo = np.max(test_volume * test_volume_rho, axis = 1)
        plt.imshow(YOZ_albedo)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_albedo_YOZ.png')
        plt.close()
        XOZ_albedo = np.max(test_volume * test_volume_rho, axis = 2)
        plt.imshow(XOZ_albedo)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_albedo_XOZ.png')
        plt.close()

    del test_input
    del test_input_batch
    del test_output
    del test_volume
    return 0

def save_model(model, global_step,i,m):
    # save model
    model_name = './model/epoch' + str(i) + 'm' + str(m) + '.pt'
    torch.save(model, model_name)
    global_step += 1
    
    return 0

def updata_lr(optimizer,args):
        # clear varible
    # update learning rate
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0000001:
            
            
            # param_group['lr'] = param_group['lr'] * 0.987
            # param_group['lr'] = param_group['lr'] * 0.993
            # param_group['lr'] = param_group['lr'] * 0.995
            # param_group['lr'] = param_group['lr'] * 0.996
            # param_group['lr'] = param_group['lr'] * 0.997
            # param_group['lr'] = param_group['lr'] * 0.998
            # param_group['lr'] = param_group['lr'] * 0.999
            param_group['lr'] = param_group['lr'] * args.lr_decay
            

            learning_rate = param_group['lr']
            print('learning rate is updated to ',learning_rate)
    return 0

def target_pdf(x,y,loss_map):
    # xv = np.linspace(0, M)
    # yv = np.linspace(0, N)

    [M, N] = loss_map.shape
    index_x = int(x / 1 * M)
    index_y = int(y / 1 * N)
    if (index_x >= M) | (index_y >= N) | (index_x <= 0) | (index_y <= 0):
        return 0
    else:
        return loss_map[index_x, index_y]

def MCMC(loss, kernel_size, sampling_num, sampling_r):
    
    [M,N] = loss.shape
    kernel = torch.ones([1,1,kernel_size,kernel_size])
    loss = loss.reshape(1,1,M,N)
    loss_map = F.conv2d(loss,kernel, padding = int((kernel_size - 1) / 2))

    loss_map = loss_map.reshape([M,N])

    mean = torch.mean(loss_map)
    std = torch.std(loss_map)
    loss_map = (loss_map - mean) / std 
    loss_map = loss_map - torch.min(loss_map)

    plt.imshow(loss_map.cpu().numpy())
    plt.colorbar()
    plt.savefig('./figure/test/initial_equal_loss_map.png')
    plt.close()

    # loss_map = torch.log(loss_map + 1)
    # loss_map = torch.exp(loss_map)
    # loss_map = loss_map ** 2

    plt.imshow(loss_map.cpu().numpy())
    plt.colorbar()
    plt.savefig('./figure/test/log_equal_loss_map.png')
    plt.close()

    loss_map = loss_map / torch.sum(loss_map)

    plt.imshow(loss_map.cpu().numpy())
    plt.colorbar()
    plt.savefig('./figure/test/pdf_equal_loss_map.png')
    plt.close()


    Samp_Num = sampling_num
    Init_Num = 100
    sampling_radius = sampling_r

    samples = np.zeros([Samp_Num + Init_Num + 1, 2])
    init = np.random.random([1,2])
    samples[0,:] = init
    q = lambda v: np.array([np.random.normal(v[0],sampling_radius ** 2), np.random.normal(v[1],sampling_radius ** 2)])
    uu = np.random.random(Samp_Num + Init_Num)
    samples_pdf = np.zeros(Samp_Num + Init_Num + 1)
    for i in range(Samp_Num + Init_Num):
        xstar = q(samples[i,:])
        samples_pdf[i] = target_pdf(samples[i,0], samples[i,1], loss_map)
        alpha = min(1,target_pdf(xstar[0], xstar[1], loss_map) / samples_pdf[i])
        if uu[i] < alpha:
            samples[i+1,:] = xstar
        else:
            samples[i+1,:] = samples[i,:]
        if i % int(Samp_Num / 10) == 0:
            print('MCMC:', i,'/',str(Samp_Num), samples[i,:])
    samples = samples[Init_Num + 1::,:]
    samples[:,0] = np.round(samples[:,0] * M)
    samples[:,1] = np.round(samples[:,1] * N)
    samples = samples.astype(np.int) - 1

    return samples

def data_rebalance(args, total_loss, total_camera_grid_positions, nlos_data, camera_grid_positions, camera_grid_size, index, device, *, total_laser_grid_positions):

    [_,M,N] = nlos_data.shape    
    total_loss = total_loss.reshape([M,N]).T
    
    plt.imshow(total_loss.cpu().numpy())
    plt.colorbar()
    plt.savefig('./figure/test/initial_equal_loss_map.png')
    plt.close()
    

    samples = MCMC(total_loss, kernel_size = 9, sampling_num = M * N, sampling_r = 0.3)

    plt.hist2d(samples[:,1], samples[:,0], bins = 25) # 由于显示图像的问题，要交换xy轴来显示
    plt.xlim(0,256)
    plt.ylim(256,0)
    plt.colorbar()
    plt.savefig('./figure/test/2Dhist_samples.png')
    plt.close()

    samples_rebalanced = samples[:,[1,0]]

    nlos_data_rebalanced = torch.zeros(nlos_data.shape).to(device)
    camera_grid_positions_rebalanced = np.zeros(camera_grid_positions.shape)
    if not args.confocal:
        laser_grid_positions_rebalanced = np.zeros(laser_grid_positions.shape)

    for hi in range(M):
        for hj in range(N):
            h = hi * N + hj
            Ni = samples_rebalanced[h, 0]
            Nj = samples_rebalanced[h, 1]
            indexmatrix = np.where(index == (Ni * N + Nj))[0]
            if indexmatrix.shape[0] == 0:
                loc = 0
            else:
                loc = indexmatrix[0]
            n = (loc % N)
            m = int((loc - n) / N)
            
            nlos_data_rebalanced[:, hi, hj] = nlos_data[:, m, n]
            camera_grid_positions_rebalanced[:, hi * N + hj] = camera_grid_positions[:, m * N + n]
            if not args.confocal:
                laser_grid_positions_rebalanced[:, hi * N + hj] = laser_grid_positions[:, m * N + n]
            # nlos_data_rebalanced[:, hi, hj] = total_nlos_data[:, Ni, Nj]
            # camera_grid_positions_rebalanced[:, h] = total_camera_grid_positions[:, Ni * N + Nj]
            if h % int(M * N / 10) == 0:
                print('Rebalance:', h,'/',str(M * N))
                # print(m)
    
    x_hist = camera_grid_positions_rebalanced[0,:]
    z_hist = camera_grid_positions_rebalanced[2,:]
    plt.hist2d(x_hist, z_hist, bins = 25) # 由于显示图像的问题，要交换xy轴来显示
    plt.xlim(- camera_grid_size[0] / 2, camera_grid_size[0] / 2)
    plt.ylim(- camera_grid_size[1] / 2, camera_grid_size[1] / 2)
    plt.colorbar()
    plt.savefig('./figure/test/distribution_camera_grid_positions.png')
    plt.close()
    # for i in range(M):
    #     for j in range(N):
    #         Ni = samples[i * N + j, 0]
    #         Nj = samples[i * N + j, 1]
    #         nlos_data_rebalanced[:,i,j] = nlos_data[:,Ni, Nj]
    #         camera_grid_positions_rebalanced[:, i * N + j] = camera_grid_positions[:, Ni * N + Nj]

    if args.confocal:
        return nlos_data_rebalanced, camera_grid_positions_rebalanced
    elif not args.confocal:
        return nlos_data_rebalanced, camera_grid_positions_rebalanced, laser_grid_positions_rebalanced

def transformer(nlos_data, args, d, device):
    
    data = nlos_data[args.start:args.end,:,:] * args.gt_times
    [L, M, N] = data.shape
    data = data.cpu().numpy().reshape([L,M * N]).T
    n = 1000
    step = max(int(M * N / n),1)
    data_part = data[np.random.permutation(M * N)[::step],:]
    n = data_part.shape[0]

    mu = np.mean(data_part, axis = 0).reshape([1,L])
    X = data_part - mu
    [U,s,Vh] = linalg.svd(X)
    S = np.zeros([U.shape[0],Vh.shape[0]])
    for r in range(s.shape[0]):
        S[r,r] = s[r]
    V = Vh.T

    # Y = X @ V
    # Y_down = X @ V[:,0:d]
    # data_down = (V[:,0:d] @ Y_down.T).T + mu
    # data_re = Y @ V.T + mu

    # samples: n x D
    transform_matrix = torch.from_numpy(V[:,0:d]).to(device) # D x d
    transform_vector = torch.from_numpy(mu.reshape([1,-1])).to(device) # n x D

    # test
    # for r in range(10):
    #     sample = nlos_data[args.start:args.end,0, r].reshape([1,-1])
    #     transformed_sample = torch.mm((sample - transform_vector), transform_matrix)
    #     sample_re = torch.mm(transform_matrix, transformed_sample.T).T + transform_vector

    #     plt.plot(sample.reshape([-1]).cpu().numpy(),linewidth=0.1,marker = 'x',markersize=1, label = 'sample')
    #     plt.plot(sample_re.reshape([-1]).cpu().numpy(),linewidth=0.1,marker = 'o',markersize=1, label = 'sample_re')
    #     plt.legend(loc='upper right')
    #     plt.savefig('./figure/test/PCA1_samples' + str(r) + '.png')
    #     plt.close()

    #     print(torch.sum((sample - sample_re) ** 2))
    #     plt.plot(transformed_sample.reshape([-1]).cpu().numpy(), label = 'feature')
    #     plt.legend(loc='upper right')
    #     plt.savefig('./figure/test/PCA2_samples' + str(r) + '.png')
    #     plt.close()

    return transform_matrix, transform_vector

def MCMC_sampling(network_res, theta_min, theta_max, phi_min, phi_max, device, camera_grid_position, c, deltaT, no_rho, start, end, args):
    # camera_grid_positions[:,m * N + n + j], args.num_sampling_points, args.test_accurate_sampling, volume_position, volume_size, c, deltaT, args.no_rho, args.start, args.end
    with torch.no_grad():
        [L,M,N] = network_res.shape

        # # network_res = (network_res > (torch.mean(network_res) + torch.std(network_res))).float()
        # projected_image = torch.max(network_res, dim = 0).values
        # # projected_image = torch.log(projected_image + 1)
        # # projected_image = torch.log(projected_image + 1)
        # plt.imshow(projected_image.cpu().numpy())
        # plt.colorbar()
        # plt.savefig('./figure/test/MCMC2/image_projected.png')
        # plt.close()

        # for k in range(network_res.shape[0]):
        #     plt.imshow(network_res[k,:,:].cpu().numpy())
        #     plt.colorbar()
        #     plt.savefig('./figure/test/MCMC2/image_' + str(k) + '.png')
        #     plt.close()



        kernel_size = int(M / 8 + 1) # 必须奇数
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = get_gaussian_kernel(kernel_size = kernel_size, kernel_radius = 2.0)
        # for i in range(kernel_size):
        #     plt.imshow(kernel[i,:,:], vmin=0, vmax=0.1)
        #     plt.colorbar()
        #     plt.savefig('./figure/test/gaussian_pdf/pdf' + str(i) + '.png')
        #     plt.close()
        kernel = torch.from_numpy(kernel).to(device).float()
        kernel = kernel.reshape([1, 1, kernel_size, kernel_size, kernel_size])

        network_res = network_res.reshape(1,1,L,M,N)
        pdf_map = F.conv3d(network_res, kernel, padding = int((kernel_size - 1) / 2))
        pdf_map = pdf_map.reshape(L,M,N)
        pdf_map_numpy = pdf_map.cpu().numpy()
        # for k in range(pdf_map.shape[0]):
        #     print(k)
        #     plt.imshow(pdf_map_numpy[k,:,:])
        #     plt.colorbar()
        #     plt.savefig('./figure/test/MCMC2/image_' + str(k) + '.png')
        #     plt.close()

        # samples = torch.zeros([L,M,N])
        # for k in range(L):
        #     samples.append(MCMC_2(pdf_map[k,:,:], 32 ** 2, 0.3, theta_min, theta_max, phi_min, phi_max, k))
        #     print(k)
        #     if (k) % int(L / 10) == 0:
        #       print('MCMC:', k ,'/',str(L))

        
        spherical_samples = np.zeros([L, args.num_MCMC_sampling_points ** 2, 2])
        samples_pdf = np.zeros([L, args.num_MCMC_sampling_points ** 2])
        cores = multiprocessing.cpu_count()
        cores = 6
        pool = multiprocessing.Pool(processes = cores)
        result_list = []
        for k in range(L):
            result_list.append(pool.apply_async(func = MCMC_2, args = (pdf_map_numpy[k,:,:], args.num_MCMC_sampling_points ** 2, 0.3, theta_min, theta_max, phi_min, phi_max, k, args)))
            # if (k) % int(L / 10) == 0:
            #     print('MCMC:', k ,'/',str(L))
        pool.close()
        pool.join()

        time1 = time.time()
        for i in range(len(result_list)):
            result = result_list[i].get()
            spherical_samples[i,:,:] = result[0]
            samples_pdf[i,:] = result[1]
        
        '''
        spherical_samples = np.zeros([L, args.num_MCMC_sampling_points ** 2, 2])
        samples_pdf = np.zeros([L,args.num_MCMC_sampling_points ** 2])
        for k in range(L):
            a, b = MCMC_2(pdf_map_numpy[k,:,:], args.num_MCMC_sampling_points ** 2, 0.3, theta_min, theta_max, phi_min, phi_max, k, args)
            spherical_samples[k,:,:] = a
            samples_pdf[k,:] = b
            # if (k) % int(L / 10) == 0:
            #     print('MCMC:', k ,'/',str(L))
        '''

        spherical_samples = torch.from_numpy(spherical_samples).float().to(device)
        samples_pdf = torch.from_numpy(samples_pdf).float().to(device)

        r_min = start * c * deltaT # zaragoza256_2 
        r_max = end * c * deltaT
        num_r = math.ceil((r_max - r_min) / (c * deltaT))
        r = torch.linspace(r_min, r_max , num_r)
        r = r.reshape(-1,1,1)
        # r = np.matlib.repmat(r, 1, M * N, 1)
        r = r.repeat(1, args.num_MCMC_sampling_points ** 2 ,1)
        
        spherical_samples = torch.cat((r, spherical_samples), axis = 2)
        spherical_samples = spherical_samples.reshape(-1,3)
        samples_pdf = samples_pdf.reshape(-1)
        cartesian_samples = spherical2cartesian_torch(spherical_samples)
        cartesian_samples = cartesian_samples + torch.from_numpy(camera_grid_position.reshape(1,3)).float().to(device)
        input_points_extra = torch.cat((cartesian_samples,spherical_samples[:, 1:]), axis = 1)

    return input_points_extra, samples_pdf

def get_gaussian_kernel(kernel_size = 15, kernel_radius = 1.0):

    x = np.linspace(-kernel_radius/2, kernel_radius/2, kernel_size)
    y = np.linspace(-kernel_radius/2, kernel_radius/2, kernel_size)
    z = np.linspace(-kernel_radius/2, kernel_radius/2, kernel_size)
    [X,Y,Z] = np.meshgrid(x,y,z)
    X = X.reshape(kernel_size ** 3, 1)
    Y = Y.reshape(kernel_size ** 3, 1)
    Z = Z.reshape(kernel_size ** 3, 1)
    coords = np.concatenate((X,Y,Z), axis = 1)

    # pdf = np.zeros([kernel_size, kernel_size, kernel_size])
    mean = np.array([0,0,0])
    cov = np.eye(3)

    # for i in range(kernel_size):
    #     for j in range(kernel_size):
    #         for k in range(kernel_size):
    #             pdf[i,j,k] = multivariate_normal.pdf(np.array([X[i,j,k],Y[i,j,k],Z[i,j,k]]), mean = mean, cov = cov )
    pdf = multivariate_normal.pdf(coords, mean = mean, cov = cov )
    pdf = pdf.reshape(kernel_size, kernel_size, kernel_size)
    # pdf = np.fft.fftshift(pdf)
    # pdf = np.roll(pdf, int(math.ceil(kernel_size / 2)), axis = 0)
    return pdf

def MCMC_2(pdf_map, sampling_num, sampling_r, theta_min, theta_max, phi_min, phi_max, k, args):


    [M,N] = pdf_map.shape
    dtheta = (theta_max - theta_min) / (M)
    dphi = (phi_max - phi_min) / (N)


    # pdf_map = (pdf_map > torch.mean(pdf_map)).float()

    mean = np.mean(pdf_map)
    std = np.std(pdf_map)
    pdf_map = (pdf_map - mean) / std 
    pdf_map = pdf_map - np.min(pdf_map)
    

    # plt.imshow(pdf_map)
    # plt.colorbar()
    # plt.savefig('./figure/test/initial_equal_pdf_map/initial_equal_pdf_map' + str(k) + '.png')
    # plt.close()

    pdf_map = np.log(pdf_map + 1)
    # pdf_map = np.exp(pdf_map)
    # pdf_map = pdf_map ** 2

    pdf_map = pdf_map + np.max(pdf_map) / 100
    # plt.imshow(pdf_map.cpu().numpy())
    # plt.colorbar()
    # plt.savefig('./figure/test/processed_equal_pdf_map/processed_equal_pdf_map' + str(k) + '.png')
    # plt.close()


    # pdf_map = np.ones([M,N])


    pdf_map = pdf_map / np.sum(pdf_map) * 1 / (dtheta * dphi)

    # plt.imshow(pdf_map)
    # plt.colorbar()
    # plt.savefig('./figure/test/pdf_equal_pdf_map/pdf_equal_pdf_map' + str(k) + '.png')
    # plt.close()


    Samp_Num = sampling_num
    Init_Num = 100
    sampling_radius = sampling_r

    samples = np.zeros([Samp_Num + Init_Num + 1, 2])
    init = np.random.random([1,2]) / 2 + 0.25
    samples[0,:] = init
    q = lambda v: np.array([np.random.normal(v[0],sampling_radius ** 2), np.random.normal(v[1],sampling_radius ** 2)])
    uu = np.random.random(Samp_Num + Init_Num)
    samples_pdf = np.zeros(Samp_Num + Init_Num + 1)
    # time0 = time.time()
    for i in range(Samp_Num + Init_Num):
        xstar = q(samples[i,:])
        # xstar = np.array([1,1])
        # if i == 0:
        #     a = time.time() - time0
        samples_pdf[i] = target_pdf(samples[i,0], samples[i,1], pdf_map)
        alpha = min(1,target_pdf(xstar[0], xstar[1], pdf_map) / samples_pdf[i])
        # p2 = target_pdf(samples[i,0], samples[i,1], pdf_map)
        # p1 = target_pdf(xstar[0], xstar[1], pdf_map)
        if uu[i] < alpha:
            samples[i+1,:] = xstar
        else:
            samples[i+1,:] = samples[i,:]
        # if (i - Init_Num) % int(Samp_Num / 10) == 0:
        #     print('MCMC:', i - Init_Num ,'/',str(Samp_Num), samples[i,:])
    # print((time.time() - time0) * 200)
    samples = samples[Init_Num + 1::,:]
    samples_pdf = samples_pdf[Init_Num + 1::]
    samples_pdf[-1] = target_pdf(samples[-1,0], samples[-1,1], pdf_map)
    # samples[:,0] = np.round(samples[:,0] * M)
    # samples[:,1] = np.round(samples[:,1] * N)
    # samples = samples.astype(np.int) - 1


    # print(dtheta, dphi)

    samples[:,0] = samples[:,0] * args.num_sampling_points * dtheta + theta_min
    samples[:,1] = samples[:,1] * args.num_sampling_points * dphi + phi_min

    # plt.hist2d(samples[:,1], samples[:,0], bins = 32) # 由于显示图像的问题，要交换xy轴来显示
    # plt.xlim(phi_min,phi_max)
    # plt.ylim(theta_max,theta_min)#norm=norm, 
    # # plt.colorbar(matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.Spectral_r))
    # plt.colorbar()
    # plt.savefig('./figure/test/2Dhistogram_samples/hist_samples_' + str(k) + '.png')
    # plt.close()


    # samples_for_show = np.concatenate((samples,samples_pdf.reshape(-1,1)), axis = 1)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(samples_for_show[:,0],samples_for_show[:,1],samples_for_show[:,2],c='r',alpha = 0.2, linewidths=0.01)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # plt.savefig('./figure/test/samples_with_pdf_show_3D.png')
    # plt.close()

    # plt.scatter(samples_for_show[:,0],samples_for_show[:,1],c='r',alpha = 0.2, linewidths=0.01)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    # plt.savefig('./figure/test/samples_with_pdf_show_XOY.png')
    # plt.close()

    # plt.scatter(samples_for_show[:,0],samples_for_show[:,2],c='r',alpha = 0.2, linewidths=0.01)
    # plt.xlabel('X')
    # plt.ylabel('Z')
    # plt.show()
    # plt.savefig('./figure/test/samples_with_pdf_show_XOZ.png')
    # plt.close()



    return samples, samples_pdf

# if __name__=='__main__': # test for encoding
#     pt = torch.rand(3)
#     coded_pt = encoding(pt, 10)
#     pass

# if __name__ == "__main__": # test for cartesian2spherical
#     x = np.array([1,0,0,1])
#     y = np.array([0,1,0,1])
#     z = np.array([0,0,1,1])
#     pt = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[1e-4,-1e-4,1]])
#     spherical_pt = cartesian2spherical(pt)
#     print(spherical_pt)
#     pass

# if __name__ == "__main__": # test for test_set_error
#     gtdir = './data/plane_volume.mat'
#     model_path = './volumeregressionmodel/test1_withPE/epoch190_withoutPE.pt'
#     volume, volume_vector = load_generated_gt(gtdir)
#     # model = torch.load('epoch190_withoutPE.pt')
#     model = Network(D_in = 60, H = 128, D_out = 1)
#     error = test_set_error(model,volume,volume_vector,1,10,batchsize = 8192)
#     pass