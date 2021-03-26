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

from load_nlos import load_nlos_data
from load_generated_data import load_generated_data
from load_generated_data import load_generated_gt
from load_born import load_born_data

seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'''
device = torch.device("cpu")
np.random.seed(0)

DEBUG = False

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
    parser.add_argument("--hiddenlayer_dim", type=int, default=100, 
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
    parser.add_argument("--gtdir", type=str, default='./data/plane_volume.mat', 
                        help='the path of groundtruth volume')
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'nlos':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_nlos_data(args.datadir)
        # volume_position = 
        # volume_size
        pmin = np.array([-0.25,-0.65,-0.25])
        pmax = np.array([0.25,-0.35,0.25])
        print('Loaded nlos')
    elif args.dataset_type == 'generated':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_generated_data(args.datadir)
        volume, volume_vector = load_generated_gt(args.gtdir)
        pmin = np.array([-0.25,-0.75,-0.25])
        pmax = np.array([0.25,-0.25,0.25])
        print('Loaded generated data')
    elif args.dataset_type == 'born':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_born_data(args.datadir)
        volume_size = 0.1
        volume_position = np.array([0.1,-0.3,0.1])
        pmin = np.array([0, -0.4, 0])
        pmax = np.array([0.2, -0.2, 0.2])
        # print(type(nlos_data), type(camera_grid_points), type(camera_grid_positions), type(camera_grid_size), type(camera_position), type(volume_position), type(volume_size)) # <class 'numpy.ndarray'> <class 'h5py._hl.dataset.Dataset'> <class 'h5py._hl.dataset.Dataset'> <class 'h5py._hl.dataset.Dataset'> <class 'h5py._hl.dataset.Dataset'>

    # if args.dataset_type == 'llff':
    #     images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
    #                                                               recenter=True, bd_factor=.75,
    #                                                               spherify=args.spherify)
    #     hwf = poses[0,:3,-1]
    #     poses = poses[:,:3,:4]
    #     print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
    #     if not isinstance(i_test, list):
    #         i_test = [i_test]

    #     if args.llffhold > 0:
    #         print('Auto LLFF holdout,', args.llffhold)
    #         i_test = np.arange(images.shape[0])[::args.llffhold]

    #     i_val = i_test
    #     i_train = np.array([i for i in np.arange(int(images.shape[0])) if
    #                     (i not in i_test and i not in i_val)])

    #     print('DEFINING BOUNDS')
    #     if args.no_ndc:
    #         near = np.ndarray.min(bds) * .9
    #         far = np.ndarray.max(bds) * 1.
            
    #     else:
    #         near = 0.
    #         far = 1.
    #     print('NEAR FAR', near, far)

    # elif args.dataset_type == 'blender':
    #     images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    #     print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    #     i_train, i_val, i_test = i_split

    #     near = 2.
    #     far = 6.

    #     if args.white_bkgd:
    #         images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    #     else:
    #         images = images[...,:3]

    # elif args.dataset_type == 'deepvoxels':

    #     images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
    #                                                              basedir=args.datadir,
    #                                                              testskip=args.testskip)

    #     print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
    #     i_train, i_val, i_test = i_split

    #     hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
    #     near = hemi_R-1.
    #     far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    # H, W, focal = hwf
    # H, W = int(H), int(W)
    # hwf = [H, W, focal]

    # if args.render_test:
    #     render_poses = np.array(poses[i_test])

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

    # Create nerf model
    # render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    # Construct our model by instantiating the class defined above
    if args.use_encoding:
        model = Network(D_in = 6 * args.encoding_dim, H = args.hiddenlayer_dim, D_out = 1)
    else:
        model = Network(D_in = 3, H = args.hiddenlayer_dim, D_out = 1)

    # model = torch.load('./epoch5000_withPE.pt',map_location=torch.device('cpu'))

    # checkpoint = torch.load("./volumeregressionmodel/test1_withPE/epoch190_withPE.pt", map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint["state_dict"])
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    # criterion = torch.nn.MSELoss(reduction='sum')

    model = model.to(device)
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion2 = torch.nn.L1Loss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # print(nlos_data.shape[0])

    global_step = 0


    [P,Q,R] = [64,64,64]
    xv = np.linspace(pmin[0],pmax[0],P)
    yv = np.linspace(pmin[1],pmax[1],Q)
    zv = np.linspace(pmin[2],pmax[2],R)
    coords = np.stack(np.meshgrid(xv, yv, zv),-1) # coords
    coords = coords.transpose([1,0,2,3])
    coords = coords.reshape([-1,3])


    # normalization
    # coords = (coords - np.min(coords, axis = 0)) / (np.max(coords, axis = 0) - np.min(coords, axis = 0))
    # occupancy = volume.reshape([-1])

    # Move testing data to GPU
    # render_poses = torch.Tensor(render_poses).to(device)

    # for i in range(0, nlos_data.shape[1], 16): 
    #     for j in range(0, nlos_data.shape[2], 16):
    #         plt.plot(np.linspace(1,nlos_data.shape[0],nlos_data.shape[0]), nlos_data[:,i,j])
    # plt.savefig('./histogram_all_points.png')
    # # 测试用代码：画出nlos_data 的histogram

    # nlos_sum_histogram = np.sum(nlos_data,axis=2)
    # nlos_sum_histogram = np.sum(nlos_sum_histogram,axis=1)
    # plt.plot(np.linspace(1,nlos_data.shape[0],nlos_data.shape[0]), nlos_sum_histogram)
    # plt.savefig('./histogram_sum_all.png')
    # # 测试用代码：画出nlos_data的所有 histogram 之和

    nlos_data = torch.Tensor(nlos_data).to(device)
    # camera_grid_positions = torch.Tensor(camera_grid_positions).to(device)

    if args.test_neglect_former_bins:
        # threshold_former_bin = threshold_bin(nlos_data)
        print('all bins < ',args.test_neglect_former_nums ,' are neglected')
    else:
        pass

    # N_iters = 200000 + 1

    N_iters = 1
    [L,M,N] = nlos_data.shape
    I = args.test_neglect_former_nums
    K = 2 # K 用 2 的倍数
    batchsize = (L - I + 1) * K # K 用 2 的倍数
    print('Training Begin')

    # [P,Q,R] = [64,64,64]
    test_batchsize = 64 * 64 * 64
    # test_batchsize = 128 * 128 * 16
    # print('TRAIN views are', i_train)
    # print('TEST views are', i_test)
    # print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    # data shuffeler
    nlos_data = nlos_data.view(L,-1)
    camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)
    full_data = torch.cat((nlos_data, camera_grid_positions), axis = 0)
    full_data = full_data[:,torch.randperm(full_data.size(1))]
    nlos_data = full_data[0:1024,:].view(L,M,N)
    camera_grid_positions = full_data[1024:,:].numpy()
    del full_data

    start = 0
    
    for i in trange(start, N_iters):
        # time0 = time.time()

        for m in range(0, M, 1):

            # save predicted volume
            # normalization

            test_input = (coords - pmin) / (pmax - pmin)
            test_input = encoding_batch_numpy(test_input, args.encoding_dim)
            test_input = torch.from_numpy(test_input).float().to(device)
            test_output = torch.empty(P * Q * R).to(device)
            with torch.no_grad():
                for l in range(int(P * Q * R / test_batchsize)):
                    test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                    test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize] = model(test_input_batch).view(-1)

            # test_gt = torch.from_numpy(occupancy).float().to(device)
            # with torch.no_grad():
            #     error = criterion(test_output, test_gt)
            test_volume = test_output.view(P,Q,R)
            test_volume = test_volume.numpy()
            mdic = {'volume':test_volume, 'x':xv, 'y':yv, 'z':zv}
            scipy.io.savemat('./model/predicted_volume' + str(i) + str(m) + '.mat', mdic)
            print('save predicted volume in epoch ' + str(i))
            del test_input
            del test_input_batch
            del test_output
            del test_volume

            # clear varible

            # update learning rate
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.0001:
                    param_group['lr'] = param_group['lr'] * 0.93
                    learning_rate = param_group['lr']
            print('learning rate is updated to ',learning_rate)

            # save model
            model_name = './model/epoch' + str(i) + 'm' + str(m) + '.pt'
            torch.save(model, model_name)
            global_step += 1

            for n in range(0, N, 1):

                time0 = time.time()
                # camera_position = camera_grid_positions[::].reshape([3,nlos_data.shape[1],nlos_data.shape[2]])[:,m,n]
                if args.test_neglect_former_bins:
                    
                    [x0,y0,z0] = [camera_grid_positions[0, m * M + n], camera_grid_positions[1, m * M + n],camera_grid_positions[2, m * M + n]]
                    input_points = coords.reshape([-1,3])
                    # show_samples(input_points, volume_position, volume_size, camera_grid_positions[:,m * M + n])
                    input_points_sph = cartesian2spherical(input_points - np.array([x0,y0,z0]))

                    sequence = np.concatenate((input_points,input_points_sph), axis=1) # 前三个元素是直角坐标系下的绝对坐标，后三个元素是球坐标系下的相对camera坐标
                    sorted_sequence = sequence[np.argsort(sequence[:,3])]
                    del sequence
                    del input_points
                    del input_points_sph

                    # normalization
                    sorted_sequence[:,0:3] = (sorted_sequence[:,0:3] - pmin) / (pmax - pmin)
                    
                    r = sorted_sequence[:,3]
                    if args.use_encoding:
                        input_points_coded = encoding_batch_numpy(sorted_sequence[:,0:3], args.encoding_dim)
                        input_points_coded = torch.from_numpy(input_points_coded).float().to(device)


                    network_res = torch.empty(P * Q * R).to(device)
                    for l in range(int(P * Q * R / test_batchsize)):
                        input_batch = input_points_coded[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                        network_res[0 + l * test_batchsize :test_batchsize + l * test_batchsize] = model(input_batch).view(-1)
                    network_res = network_res.view(-1)

                    # network_res = network_res * (network_res > 0.5) # 去volume噪声处理
                    # network_res = network_res.view(P,Q,R)
                    # network_res = network_res.permute(0,2,1)
                    # network_res = network_res.contiguous().view(-1) # transpose 处理

                    # with torch.no_grad():
                    #     networkvolume = test_output.view(P,Q,R)
                    #     networkvolume = networkvolume.numpy()
                    #     ndic = {'volume':networkvolume, 'x':xv, 'y':yv, 'z':zv}
                    #     scipy.io.savemat('./model/network_volume.mat', ndic)

                    pred_histgram = torch.zeros(L - I).to(device)
                    
                    for l in range(L - I):
                        distancemin = (l + I) * 4e-12 * 3e8
                        distancemax = (l + I + 1) * 4e-12 * 3e8
                        startindex = np.searchsorted(r,distancemin)
                        endindex = np.searchsorted(r,distancemax)
                        pred_histgram[l] = torch.sum(network_res[startindex:endindex]) / (distancemax ** 4)
                        # print(startindex,endindex)
                        # pred_histgram[l] = torch.sum(network_res * torch.from_numpy((r > distancemin) * (r < distancemax)).float().to(device)) / (distancemax ** 4) # 距离衰减
                        # print(l,distancemin,distancemax)

                        # print(pred_histgram[l] - torch.sum(network_res * torch.from_numpy((r > distancemin) * (r < distancemax))).float().to(device) / (distancemax ** 4))
                        
                    nlos_histogram = nlos_data[I:, m, n]

                    '''
                    input_points = coords
                    input_points = input_points.reshape([-1,3])

                    # nomalization, 考虑到读取模型和预训练，这里的归一化方法与shape regression保持一致
                    pmin = np.array([-0.25,-0.75,-0.25])
                    pmax = np.array([0.25,-0.25,0.25])
                    input_points = (input_points  - pmin) / (pmax - pmin)
                    
                    # input_points = torch.from_numpy(input_points).float()
                    if args.use_encoding:
                        input_points = encoding_batch_numpy(input_points, args.encoding_dim).reshape([-1, 3 * args.encoding_dim * 2])
                    network_res = torch.empty((L - I + 1) * (args.num_sampling_points ** 2))
                    # 原size ([L - I + 1,args.num_sampling_points ** 2, 3 * args.encoding_dim * 2]), reshape 成 [-1,3 * args.encoding_dim * 2]
                    # input_points = torch.Tensor(input_points)
                    # input_points = torch.Tensor(input_points).to(device)
                    for k in range(int(args.num_sampling_points ** 2 / K)):
                        input_batch = input_points[0 + k * batchsize:batchsize + k * batchsize,:]
                        input_batch = torch.Tensor(input_batch).to(device)
                        network_res[(0 + k * batchsize):(batchsize + k * batchsize)] = model(input_batch).view(-1)
                    network_res = network_res.to(device)
                    network_res = network_res.view(L - I + 1, args.num_sampling_points ** 2)
                    distance = torch.from_numpy((np.linspace(I,L,L - I + 1) * 4 * 1e-12 * 3 * 1e8))
                    network_res = torch.sum(network_res, axis = 1)
                    network_res = network_res # / (distance ** 4)

                    nlos_histogram = nlos_data[(args.test_neglect_former_nums - 1):, m, n]'''
                else:
                    # input_hist = [spherical_sample((k+1)*4, camera_grid_positions[0, m*M+n], camera_grid_positions[1, k], camera_grid_positions[2, m*M+n], args.test_accurate_sampling, volume_position, volume_size) for k in range(nlos_data.shape[2])]
                    # if args.use_encoding:
                    #     input_hist = [encoding_sph(input_hist[k], args.encoding_dim) for k in range(len(input_hist))]
                    # input_hist = torch.Tensor(input_hist)
                    # input_hist = torch.Tensor(input_hist).to(device)
                    # network_res = torch.cat([model(input_hist[k]) for k in range(nlos_data.shape[2])], 1)
                    # network_res = torch.sum(network_res, 1)
                    # for k in range(nlos_data.shape[2]):
                    #     network_res[k] = torch.div(network_res[k], math.pow(( 4 * (k+1) * 3 / 100), 2))                   
                    # nlos_histogram = nlos_data[m, n, :]
                    pass


                # data post processing
                # nlos_histogram = (nlos_histogram - torch.mean(nlos_histogram)) / (torch.var(nlos_histogram) ** 0.5)
                # network_res = (network_res - torch.mean(network_res)) / (torch.var(network_res) ** 0.5)

                # data post processing
                # nlos_histogram = (nlos_histogram - torch.min(nlos_histogram)) / (torch.max(nlos_histogram) - torch.min(nlos_histogram))
                # pred_histgram = (pred_histgram - torch.min(pred_histgram)) / (torch.max(pred_histgram) - torch.min(pred_histgram))

                # data post processing 顺序重要，不能换
                # pred_histgram = (pred_histgram - torch.min(nlos_histogram)) / (torch.max(nlos_histogram) - torch.min(nlos_histogram))
                # nlos_histogram = (nlos_histogram - torch.min(nlos_histogram)) / (torch.max(nlos_histogram) - torch.min(nlos_histogram))

                pred_histgram = pred_histgram
                nlos_histogram = nlos_histogram * 25000


                plt.plot(nlos_histogram.cpu(), alpha = 0.5, label = 'data')
                plt.plot(pred_histgram.cpu().detach().numpy(), alpha = 0.5, label='predicted')
                plt.legend(loc='upper right')
                plt.savefig('./figure/' + str(i) + str(m) + str(n) + 'histogram')
                plt.close()

                # print(samples)
                # 8.19 周三早上对 samples 的数据ad类型做了改变，原来是二重 list， 现在改成了 np 二维数组，便于向量化处理

                # if nlos_histogram > 0.0001:
                    # show_samples(samples, volume_position, volume_size) 
                    # pass
                # 测试用代码：在3d图中显示samples，并保存在当前目录里

                """  sphere_res = 0 # sphere_res 是nerf渲染出的单个bin的结果， 是一个torch.Tensor标量
                for l in range(len(samples[0])):
                x = samples[0][l]
                y = samples[1][l]
                z = samples[2][l]
                pt = torch.tensor([x, y, z], dtype=torch.float32)
                if args.use_encoding:
                    coded_pt = encoding(pt, L=args.encoding_dim) # encoding 函数将长度为 3 的 tensor 返回为长度为 6L 的tensor
                    network_res = model(coded_pt)
                else:
                    network_res = model(pt)
                sphere_res = sphere_res + network_res[0]  """ # 渲染这一部分用一个函数 nlos_render 来实现

                optimizer.zero_grad()
                loss = criterion(pred_histgram, nlos_histogram) # + 0.5 * criterion2(network_res, torch.zeros(network_res.shape).to(device)) # 注意：loss 函数内部已经集成了平均值处理
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                dt = time.time()-time0
                
                print(i,'/',N_iters,'iter  ', m,'/',nlos_data.shape[1],'  ', n,'/',nlos_data.shape[2], '  histgram loss: ',loss , 'time: ', dt)

            





if __name__=='__main__':
    # python run_nerf.py --config configs/nlos.txt
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
