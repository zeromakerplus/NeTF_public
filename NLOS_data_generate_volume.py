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

# nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_nlos_data('./data//bunny_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.21]_s[64]_l[64]_gs[0.60]_conf.mat')
# del nlos_data
camera_position = [0, -0.125, 0]
camera_grid_size = [1,1] # 这里应该是指墙上grid的长度是 1m x 1m, 换言之[-0.5, +0.5]
volume_position = np.array([0, -0.5, 0])
volume_size = 0.2 # Zaragoza Dataset 中 256 的数据里，物体大小是0.8
camera_grid_points = np.array([256,256])

[M,N] = camera_grid_points.astype(np.int)
xv_camera = np.linspace(-0.5, 0.5, 256)
yv_camera = np.linspace(0, 0, 256)
zv_camera = np.linspace(-0.5, 0.5, 256)
camera_grid_positions = np.stack(np.meshgrid(xv_camera, yv_camera, zv_camera),-1)
camera_grid_positions = camera_grid_positions.transpose([1,0,2,3])
camera_grid_positions = camera_grid_positions[:,0,:,:]
camera_grid_positions = camera_grid_positions.reshape(-1,3).T

L = 1024

datafile = './data/bunny_volume_128_oneside.mat'
datadic = scipy.io.loadmat(datafile)
volume = datadic['Volume'].astype(np.float)
xv = datadic['x'].reshape([-1])
yv = datadic['y'].reshape([-1])
zv = datadic['z'].reshape([-1])
[P,Q,R] = volume.shape
coords = np.stack(np.meshgrid(xv, yv, zv),-1) # coords
coords = coords.transpose([1,0,2,3])
coords = coords.reshape([-1,3])


# normalization
# coords = (coords - np.min(coords, axis = 0)) / (np.max(coords, axis = 0) - np.min(coords, axis = 0))
occupancy = volume.reshape([-1])
nlos_histgram = np.zeros([L,M,N])
for m in range(0, M, 1):
    for n in range(0, N, 1):

        time0 = time.time()
        # print(camera_grid_positions[:, m * M + n])
        
            
        [x0,y0,z0] = [camera_grid_positions[0, m * M + n], camera_grid_positions[1, m * M + n],camera_grid_positions[2, m * M + n]]
        input_points = coords.reshape([-1,3])
        input_points_sph = cartesian2spherical(input_points - np.array([x0,y0,z0]))
        # print(time.time() - time0)

        sequence = np.concatenate((input_points,input_points_sph, occupancy.reshape([-1,1])), axis=1) # 前三个元素是直角坐标系下的绝对坐标，后三个元素是球坐标系下的相对camera坐标
        sorted_sequence = sequence[np.argsort(sequence[:,3])]
        # print(time.time() - time0)
        # # normalization
        # pmin = np.array([-0.25,-0.75,-0.25])
        # pmax = np.array([0.25,-0.25,0.25])
        # sorted_sequence[:,0:3] = (sorted_sequence[:,0:3] - pmin) / (pmax - pmin)

        r = sorted_sequence[:,3]

        for l in range(L):
            distancemin = l * 4e-12 * 3e8
            distancemax = (l + 1) * 4e-12 * 3e8
            # nlos_histgram[l,m,n] = np.sum(sorted_sequence[:,6] * (r > distancemin) * (r < distancemax)) / (distancemax ** 4) # 距离衰减

            startindex = np.searchsorted(r,distancemin)
            # endindex = np.searchsorted(r,distancemax)
            endindex = np.searchsorted(r[startindex:(startindex + 256 * 256 * 10)],distancemax) + startindex
            # print(startindex, endindex, endindex2)
            nlos_histgram[l,m,n] = np.sum(sorted_sequence[startindex:endindex,6]) / (distancemax ** 4)
            # print(a - nlos_histgram[l,m,n])
            # print(l,distancemin,distancemax)

        dt = time.time()-time0

        print(m,'/',M,' ',n,'/',N, ' time:', dt)
        plt.plot(nlos_histgram[:,m,n])
        plt.savefig('./datageneration/' + str(m) + '_'+ str(n) + 'data_histogram')
        plt.close()



        
                
mdic = {"data": nlos_histgram, \
    "cameraPosition": camera_position, \
    "cameraGridSize": camera_grid_size, \
    "cameraGridPositions": camera_grid_positions, \
    "cameraGridPoints": camera_grid_points,\
    "hiddenVolumePosition": volume_position,\
    "hiddenVolumeSize": volume_size }

scipy.io.savemat("./datageneration/NLOS_data_generated_volume.mat", mdic)

# 最后用matlab平滑处理一下


