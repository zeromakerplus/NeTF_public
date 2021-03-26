import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import scipy.io

import matplotlib.pyplot as plt


from run_nerf_helpers import *
from MLP import *

from load_nlos import *
from math import ceil

seed = 76
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
np.random.seed(seed)

DEBUG = False

'''
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_nlos(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

    
def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret
'''

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    # NeRF-NLOS arguments
    parser.add_argument("--use_encoding", action='store_true', 
                        help='use positional encoding or not')
    # parser.add_argument("--num_layer", type=int, default=2, 
    #                     help='the number of layers in MLP')
    parser.add_argument("--hiddenlayer_dim", type=int, default=64, 
                        help='the dimmension of hidden layer')                    
    parser.add_argument("--encoding_dim", type=int, default=10, 
                        help='the dimmension of positional encoding, also L in the paper, attention that R is mapped to R^2L')
    parser.add_argument("--test_neglect_former_bins", action='store_true', 
                        help='when True, those former histogram bins will be neglected and not used in optimization. The threshold is computed automatically to ensure that neglected bins are zero')
    parser.add_argument("--test_neglect_former_nums", type=int, default=0, 
                        help='nums of values ignored')
    parser.add_argument("--test_accurate_sampling", action='store_true', 
                        help='when True, the sampling function will sample from the known object box area, rather than the whole function')
    parser.add_argument("--num_sampling_points", type=int, default=16, 
                        help='number of sampling points in one direction, so the number of all sampling points is the square of this value')
    parser.add_argument("--load_groundtruth_volume", action='store_true', 
                        help='load groundtruth volume or not')
    parser.add_argument("--no_rho", action='store_true', 
                        help='network no_rho or not')  
    # parser.add_argument("--gtdir", type=str, default='./data/plane_volume.mat', 
    #                     help='the path of groundtruth volume')
    parser.add_argument("--hierarchical_sampling", action='store_true', 
                        help='hierarchical sampling or not')
    parser.add_argument("--cuda", type=int, default=0, 
                        help='the number of cuda')
    parser.add_argument("--histogram_batchsize", type=int, default=1, 
                        help='the batchsize of histogram')
    parser.add_argument("--start", type=int, default=100, 
                        help='the start point of histogram')
    parser.add_argument("--end", type=int, default=300, 
                        help='the end point of histogram')
    parser.add_argument("--attenuation", action='store_true', 
                        help='attenuation or not')
    parser.add_argument("--two_stage", action='store_true', 
                        help='two stage learning or not')
    parser.add_argument("--lr_decay", type=float,
                        default=0.995, help='learning rate decay')
    parser.add_argument("--gt_times", type=float,
                        default=100, help='learning rate decay')
    parser.add_argument("--save_fig", action='store_true', 
                        help='save figure or not')
    parser.add_argument("--target_volume_size", type=int, default=64, 
                        help='volume size when save reconstructed volume')
    parser.add_argument("--PCA", action='store_true', 
                        help='PCA or not')
    parser.add_argument("--PCA_dim", type=int, default=256, 
                        help='PCA dimension')
    parser.add_argument("--new_model", action='store_true', 
                        help='when use two stage stargegy, take a new model or not')
    parser.add_argument("--first_stage_epoch", type=int, default=1, 
                        help='first stage epoch')
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    torch.cuda.set_device(args.cuda)
    # Load data
    if args.dataset_type == 'nlos':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_nlos_data(args.datadir)
        # volume_position = 
        # volume_size
        pmin = np.array([-0.25,-0.65,-0.25])
        pmax = np.array([0.25,-0.35,0.25])
    elif args.dataset_type == 'generated':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_generated_data(args.datadir)
        
        # volume, volume_vector = load_generated_gt(args.gtdir)
        pmin = np.array([-0.25,-0.75,-0.25])
        pmax = np.array([0.25,-0.25,0.25])
    elif args.dataset_type == 'born':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_born_data(args.datadir)
        volume_size = 0.1
        volume_position = np.array([0.1,-0.3,0.1])
        pmin = np.array([0, -0.4, 0])
        pmax = np.array([0.2, -0.2, 0.2])
        # print(type(nlos_data), type(camera_grid_points), type(camera_grid_positions), type(camera_grid_size), type(camera_position), type(volume_position), type(volume_size)) # <class 'numpy.ndarray'> <class 'h5py._hl.dataset.Dataset'> <class 'h5py._hl.dataset.Dataset'> <class 'h5py._hl.dataset.Dataset'> <class 'h5py._hl.dataset.Dataset'>
    elif args.dataset_type == 'zaragoza256':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_zaragoza256_data(args.datadir)
        # volume_position = 
        # volume_size
        # volume_position = np.array([0,-0.8,-0.4])
        # volume_size = 1.6

        pmin = volume_position - volume_size / 2
        
        pmax = volume_position + volume_size / 2
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
        # pmin = np.array([-0.5,-0.8,-0.5])
        # pmax = np.array([0.5,-0.2,0.5])
    elif args.dataset_type == 'fk':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_fk_data(args.datadir)
        # volume_position = 
        # volume_size
        pmin = volume_position - volume_size / 2
        pmax = volume_position + volume_size / 2
        # volume_size = 2
        # pmin = np.array([-1, -1.75, -1])
        # pmax = np.array([1,-0.95,1])
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
        # pmin = np.array([-0.5,-0.8,-0.5])
        # pmax = np.array([0.5,-0.2,0.5])
    elif args.dataset_type == 'specular':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_specular_data(args.datadir)
        pmin = volume_position - volume_size / 2
        pmax = volume_position + volume_size / 2
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
    elif args.dataset_type == 'lct':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_lct_data(args.datadir)
        pmin = volume_position - volume_size / 2
        pmax = volume_position + volume_size / 2
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
    # elif args.dataset_type == 'zaragoza64_raw':
    #     nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_zaragoza64_raw(args.datadir)
    #     # volume_position = 
    #     # volume_size
    #     pmin = np.array([-0.15,-0.65,-0.15])
    #     pmax = np.array([0.15,-0.35,0.15])
    #     print('Loaded nlos')
    # elif args.dataset_type == 'zaragoza256_raw':
    #     nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_zaragoza256_raw(args.datadir)
    #     # volume_position = 
    #     # volume_size
    #     pmin = np.array([-0.15,-0.65,-0.15])
    #     pmax = np.array([0.15,-0.35,0.15])
    #     print('Loaded nlos')

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    print('----------------------------------------------------')
    print('Loaded: ' + args.datadir)
    print('dataset_type: ' + args.dataset_type)
    print('gt_times: ' + str(args.gt_times))
    print('two_stage: ' + str(args.two_stage))
    print('save_fig: ' + str(args.save_fig))
    print('cuda: ' + str(args.cuda))
    print('lr_decay: ' + str(args.lr_decay))
    print('hierarchical_sampling: ' + str(args.hierarchical_sampling))
    print('accurate_sampling: ' + str(args.test_accurate_sampling))
    print('start: ' + str(args.start))
    print('end: ' + str(args.end))
    print('num_sampling_points: ' + str(args.num_sampling_points))
    print('PCA_dim: ' + str(args.PCA_dim))
    print('histogram_batchsize: ' + str(args.histogram_batchsize))
    print('target_volume_size: ' + str(args.target_volume_size))
    print('Attenuation: ' + str(args.attenuation))
    print('no_rho: ' + str(args.no_rho))
    print('----------------------------------------------------')

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    extrapath = './model/'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)
    extrapath = './figure/'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)
    extrapath = './figure/test'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)

    # Create nerf model
    # render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    # Construct our model by instantiating the class defined above
    # if args.use_encoding:
    #     model = Network_S(D_in = 6 * args.encoding_dim, H = args.hiddenlayer_dim, D_out = 1)
    # else:
    #     model = Network_S(D_in = 3, H = args.hiddenlayer_dim, D_out = 1)

    if args.use_encoding: 
        #     def __init__(self, D=8, H=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], no_rho=False)
        # model = Network(D=8, D_in = 6 * args.encoding_dim, H = args.hiddenlayer_dim, D_out = args.output_dim, no_rho=args.no_rho)
        model = Network_S_Relu(D = 8, H = args.hiddenlayer_dim, input_ch = 6 * args.encoding_dim, input_ch_views = 4 * args.encoding_dim, output_ch = 1, skips=[4], no_rho = args.no_rho)
    else:
        # model = Network(D_in = 3, H = args.hiddenlayer_dim, D_out = args.output_dim, no_rho=args.no_rho)
        model = Network_S_Relu(D = 8, H = args.hiddenlayer_dim, input_ch = 3, input_ch_views = 2, output_ch = 1, skips=[4], no_rho=args.no_rho)
    
    # model = torch.load('./model/11.28test_#156/epoch3m212.pt', map_location = 'cuda:' + str(args.cuda))
    # model = torch.load('./model/11.26test_#131/epoch3m23.pt', map_location = 'cuda:' + str(args.cuda)) # 
    # model = torch.load('./model/11.21test_#121/epoch3m54.pt', map_location = 'cuda:' + str(args.cuda))
    # model = torch.load('./model/10.27test_#50/epoch1m0.pt', map_location = 'cuda:' + str(args.cuda))
    # model = torch.load('./model/11.15test_#115/epoch2m135.pt', map_location = 'cuda:' + str(args.cuda))
    # model = torch.load('./model/10.25test_#45/epoch1m150.pt', map_location='cuda:' + str(args.cuda))
    # model = torch.load('./model/10.29test_#63/epoch1m255.pt', map_location='cuda:' + str(args.cuda))
    # 预训练模型
    if args.use_encoding: 
        #     def __init__(self, D=8, H=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], no_rho=False)
        # model = Network(D=8, D_in = 6 * args.encoding_dim, H = args.hiddenlayer_dim, D_out = args.output_dim, no_rho=args.no_rho)
        model_new = Network_S_Relu(D = 8, H = args.hiddenlayer_dim, input_ch = 6 * args.encoding_dim, input_ch_views = 4 * args.encoding_dim, output_ch = 1, skips=[4], no_rho = args.no_rho)
    else:
        # model = Network(D_in = 3, H = args.hiddenlayer_dim, D_out = args.output_dim, no_rho=args.no_rho)
        model_new = Network_S_Relu(D = 8, H = args.hiddenlayer_dim, input_ch = 3, input_ch_views = 2, output_ch = 1, skips=[4], no_rho=args.no_rho)

    # checkpoint = torch.load("./volumeregressionmodel/test1_withPE/epoch190_withPE.pt", map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint["state_dict"])
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    # criterion = torch.nn.MSELoss(reduction='sum')

    model = model.to(device)
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion2 = torch.nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # print(nlos_data.shape[0])

    global_step = 0

    target_volume_size = args.target_volume_size
    [P,Q,R] = [target_volume_size,target_volume_size,target_volume_size]
    xv = np.linspace(pmin[0],pmax[0],P)
    yv = np.linspace(pmin[1],pmax[1],Q)
    zv = np.linspace(pmin[2],pmax[2],R)
    coords = np.stack(np.meshgrid(xv, yv, zv),-1) # coords
    coords = coords.transpose([1,0,2,3])
    coords = coords.reshape([-1,3])
    if not args.no_rho:
        view_direction = np.zeros([P*Q*R, 2])
        view_direction[:,0] = np.pi / 2
        view_direction[:,1] = - np.pi / 2
        coords = np.concatenate((coords, view_direction), axis = 1)

    # pmin = volume_position - volume_size / 2
    # pmax = volume_position + volume_size / 2
    # if not args.no_rho:
    #     pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
    #     pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
    # normalization

    # for i in range(0, nlos_data.shape[1], 16): 
    #     for j in range(0, nlos_data.shape[2], 16):
    #         plt.plot(np.linspace(1,nlos_data.shape[0],nlos_data.shape[0]), nlos_data[:,i,j])
    # plt.savefig('./test/histogram_all_points.png')
    # # 测试用代码：画出nlos_data 的histogram

    # nlos_sum_histogram = np.sum(nlos_data,axis=2)
    # nlos_sum_histogram = np.sum(nlos_sum_histogram,axis=1)
    # plt.plot(np.linspace(1,nlos_data.shape[0],nlos_data.shape[0]), nlos_sum_histogram)
    # plt.savefig('./histogram_sum_all.png')
    # # 测试用代码：画出nlos_data的所有 histogram 之和

    with torch.no_grad():
        nlos_data = torch.Tensor(nlos_data).to(device)

    if args.test_neglect_former_bins:
        # threshold_former_bin = threshold_bin(nlos_data)
        print('all bins < ',args.test_neglect_former_nums ,' are neglected')
    else:
        pass

    N_iters = 10
    [L,M,N] = nlos_data.shape
    I = args.test_neglect_former_nums
    K = 2 # K 用 2 的倍数
    batchsize = (L - I + 1) * K # K 用 2 的倍数
    test_batchsize = 64 * 64 * 32
    train_batchsize = 64 * 64 * 256
    histogram_batchsize = args.histogram_batchsize
    print('Training Begin')

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    # data shuffeler
    with torch.no_grad():
        nlos_data = nlos_data.reshape(L,-1)
        camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)
        index = torch.linspace(0, M * N - 1, M * N).reshape(1, -1).float().to(device)
        full_data = torch.cat((nlos_data, camera_grid_positions, index), axis = 0)
        full_data = full_data[:,torch.randperm(full_data.size(1))]
        nlos_data = full_data[0:L,:].view(L,M,N)
        camera_grid_positions = full_data[L:-1,:].cpu().numpy()
        index = full_data[-1,:].cpu().numpy().astype(np.int)
        del full_data

    [transform_matrix, transform_vector] = transformer(nlos_data, args, d = args.PCA_dim, device = device)

    start = 0
    
    # average_loss = torch.zeros(M)
    s2 = torch.randn(1,1,100).float().to(device)
    # s2 = torch.from_numpy(np.array([1,-1]).reshape(1,1,2)).float().to(device)
    
    current_nlos_data = nlos_data
    current_camera_grid_positions = camera_grid_positions
    for i in trange(start, N_iters):
        if args.two_stage:
            if i > args.first_stage_epoch:
            # if True:
                if args.new_model:
                    if i == (args.first_stage_epoch + 1):
                        model = model_new
                
                # load_data = scipy.io.loadmat('./model/loss_2_11.19test_2.mat')
                # total_loss = torch.from_numpy(load_data['loss']).float().to(device)
                # total_camera_grid_positions = load_data['camera_grid_positions']
                scipy.io.savemat('./model/loss_' + str(i) + '.mat', {'loss':total_loss.cpu().detach().numpy(), 'camera_grid_positions': total_camera_grid_positions})
                print('save loss')

                # for p in range(256 * 256):
                #         error_l = np.sum(total_camera_grid_positions[:,index[p]] - camera_grid_positions[:,p])
                #         print(error_l)

                [nlos_data_rebalanced, camera_grid_positions_rebalanced] = data_rebalance(total_loss, total_camera_grid_positions, nlos_data, camera_grid_positions, camera_grid_size, index, device)
                
                with torch.no_grad():
                    nlos_data_rebalanced = nlos_data_rebalanced.reshape(L,-1)
                    camera_grid_positions_rebalanced = torch.from_numpy(camera_grid_positions_rebalanced).float().to(device)
                    index_rebalanced = torch.linspace(0, M * N - 1, M * N).reshape(1, -1).float().to(device)
                    full_data_rebalanced = torch.cat((nlos_data_rebalanced, camera_grid_positions_rebalanced, index_rebalanced), axis = 0)
                    full_data_rebalanced = full_data_rebalanced[:,torch.randperm(full_data_rebalanced.size(1))]
                    nlos_data_rebalanced = full_data_rebalanced[0:L,:].view(L,M,N)
                    camera_grid_positions_rebalanced = full_data_rebalanced[L:-1,:].cpu().numpy()
                    index_rebalanced = full_data_rebalanced[-1,:].cpu().numpy().astype(np.int)
                    del full_data_rebalanced
                
                current_nlos_data = nlos_data_rebalanced
                current_camera_grid_positions = camera_grid_positions_rebalanced

                stage = 'learn'
            elif (i < args.first_stage_epoch):

                stage = 'learn'
            elif (i == args.first_stage_epoch):
                stage = 'compute'
                total_loss = torch.zeros(M * N)
                total_camera_grid_positions = np.zeros(camera_grid_positions.shape)
                current_nlos_data = nlos_data
                current_camera_grid_positions = camera_grid_positions

                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
                    learning_rate = param_group['lr']
                    print('learning rate is updated to ',learning_rate)
            else:
                pass
        else:
            stage = 'learn'
            current_nlos_data = nlos_data
            current_camera_grid_positions = camera_grid_positions
        print(i,'/',N_iters,'  stage:',stage)
        # i = 1
        
        # for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.001
        #         learning_rate = param_group['lr']
        #         print('learning rate is updated to ',learning_rate)

        # time0 = time.time()

        for m in range(0, M, 1):
            if (m % 16) == 0:
                save_volume(model,coords, pmin, pmax,args, P,Q,R,device,test_batchsize,xv,yv,zv,i,m)
            save_model(model, global_step,i,m)
            if stage == 'learn':
                updata_lr(optimizer,args)
            # for n in range(0, N, histogram_batchsize):
            for n in range(0, N, 1):
                # with torch.no_grad():
                time0 = time.time()
                optimizer.zero_grad()

                for j in range(0, histogram_batchsize, 1):
                    if stage == 'learn':
                        [loss, equal_loss] = compute_loss(M,m,N,n,i,j,I,L,pmin,pmax,device,criterion,model,args,current_nlos_data,volume_position, volume_size, c, deltaT, current_camera_grid_positions,s2,transform_matrix, transform_vector)
                        loss.backward()
                        optimizer.step()

                    else:
                        with torch.no_grad():
                            [loss, equal_loss] = compute_loss(M,m,N,n,i,j,I,L,pmin,pmax,device,criterion,model,args,current_nlos_data,volume_position, volume_size, c, deltaT, current_camera_grid_positions,s2,transform_matrix, transform_vector)
                            total_loss[index[m * N + n]] = equal_loss.item()
                            total_camera_grid_positions[:,index[m * N + n]] = camera_grid_positions[:,m * N + n]

                optimizer.zero_grad()

                # 

                dt = time.time()-time0
                # print(time.time()-time0)
                if (n % 16 == 0):
                    print(i,'/',N_iters,'iter  ', m,'/', current_nlos_data.shape[1],'  ', n,'/',current_nlos_data.shape[2], '  histgram loss: ',loss.item(), 'time: ', dt)
    
    # total_loss = total_loss.reshape(M,N)
    # mdic = {'loss':total_loss.cpu().detach().numpy()}
    # scipy.io.savemat('./model/loss.mat', mdic)
    # print('save loss')

        # test_input = torch.from_numpy(coords).float().to(device)
        # with torch.no_grad():
        #     test_output = model(test_input).view(-1)
        # test_gt = torch.from_numpy(occupancy).float().to(device)
        # # error = torch.sum(torch.abs(test_output - test_gt)) / (M * N * L / skipstep / skipstep)
        # with torch.no_grad():
        #     error = criterion(test_output, test_gt)
        # test_volume = test_output.view(64,64,64)
        # test_volume = test_volume.numpy()
        # mdic = {'volume':test_volume, 'x':xv, 'y':yv, 'z':zv}
        # scipy.io.savemat('./model/predicted_volume' + str(i) + str(m) + str(n)  + '.mat', mdic)
        # print('save predicted volume in epoch ' + str(i))

        # model_name = './model/epoch' + str(i) + '.pt'
        # torch.save(model, model_name)

        # # Sample random ray batch
        # if use_batching:
        #     # Random over all images
        #     batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
        #     batch = torch.transpose(batch, 0, 1)
        #     batch_rays, target_s = batch[:2], batch[2]

        #     i_batch += N_rand
        #     if i_batch >= rays_rgb.shape[0]:
        #         print("Shuffle data after an epoch!")
        #         rand_idx = torch.randperm(rays_rgb.shape[0])
        #         rays_rgb = rays_rgb[rand_idx]
        #         i_batch = 0

        # else:
        #     # Random from one image
        #     img_i = np.random.choice(i_train)
        #     target = images[img_i]
        #     pose = poses[img_i, :3,:4]

        #     if N_rand is not None:
        #         rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

        #         if i < args.precrop_iters:
        #             dH = int(H//2 * args.precrop_frac)
        #             dW = int(W//2 * args.precrop_frac)
        #             coords = torch.stack(
        #                 torch.meshgrid(
        #                     torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
        #                     torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
        #                 ), -1)
        #             if i == start:
        #                 print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
        #         else:
        #             coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

        #         coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        #         select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        #         select_coords = coords[select_inds].long()  # (N_rand, 2)
        #         rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        #         rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        #         batch_rays = torch.stack([rays_o, rays_d], 0)
        #         target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
        #                                         verbose=i < 10, retraw=True,
        #                                         **render_kwargs_train)

        # optimizer.zero_grad()
        # img_loss = img2mse(rgb, target_s)
        # trans = extras['raw'][...,-1]
        # loss = img_loss
        # psnr = mse2psnr(img_loss)

        # if 'rgb0' in extras:
        #     img_loss0 = img2mse(extras['rgb0'], target_s)
        #     loss = loss + img_loss0
        #     psnr0 = mse2psnr(img_loss0)

        # loss.backward()
        # optimizer.step()

        # # NOTE: IMPORTANT!
        # ###   update learning rate   ###
        # decay_rate = 0.1
        # decay_steps = args.lrate_decay * 1000
        # new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lrate
        # ################################

        # dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # if i%args.i_weights==0:
        #     path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
        #     torch.save({
        #         'global_step': global_step,
        #         'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
        #         'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, path)
        #     print('Saved checkpoints at', path)

        # if i%args.i_video==0 and i > 0:
        #     # Turn on testing mode
        #     with torch.no_grad():
        #         rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
        #     print('Done, saving', rgbs.shape, disps.shape)
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        #     # if args.use_viewdirs:
        #     #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
        #     #     with torch.no_grad():
        #     #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
        #     #     render_kwargs_test['c2w_staticcam'] = None
        #     #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # if i%args.i_testset==0 and i > 0:
        #     testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
        #     os.makedirs(testsavedir, exist_ok=True)
        #     print('test poses shape', poses[i_test].shape)
        #     with torch.no_grad():
        #         render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
        #     print('Saved test set')


    
        # if i%args.i_print==0:
        #     tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        # """
        #     print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
        #     print('iter time {:.05f}'.format(dt))

        #     with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
        #         tf.contrib.summary.scalar('loss', loss)
        #         tf.contrib.summary.scalar('psnr', psnr)
        #         tf.contrib.summary.histogram('tran', trans)
        #         if args.N_importance > 0:
        #             tf.contrib.summary.scalar('psnr0', psnr0)


        #     if i%args.i_img==0:

        #         # Log a rendered validation view to Tensorboard
        #         img_i=np.random.choice(i_val)
        #         target = images[img_i]
        #         pose = poses[img_i, :3,:4]
        #         with torch.no_grad():
        #             rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
        #                                                 **render_kwargs_test)

        #         psnr = mse2psnr(img2mse(rgb, target))

        #         with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

        #             tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
        #             tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
        #             tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

        #             tf.contrib.summary.scalar('psnr_holdout', psnr)
        #             tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


        #         if args.N_importance > 0:

        #             with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
        #                 tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
        #                 tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
        #                 tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        # """

        # global_step += 1


if __name__=='__main__':
    # python run_nerf.py --config configs/nlos.txt
    # python run_nerf_batchify.py --config configs/nlos.txt
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
