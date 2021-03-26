import os
import numpy as np
import torch
import scipy.io as scio
# import h5py
 

def load_born_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data']
    data = data[:, :, :]
    # data = torch.from_numpy(data)

    camera_position = nlos_data['cameraPosition'].reshape([-1])
    camera_grid_size = nlos_data['cameraGridSize'].reshape([-1])
    camera_grid_positions = nlos_data['cameraGridPositions']
    camera_grid_points = nlos_data['cameraGridPoints'].reshape([-1])
    volume_position = nlos_data['hiddenVolumePosition'].reshape([-1])
    volume_size = nlos_data['hiddenVolumeSize'].item()

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size

def load_generated_gt(gtdir):
    volume_gt = scio.loadmat(gtdir)

    volume = volume_gt['Volume']
    xv = volume_gt['x'].reshape([-1])
    yv = volume_gt['y'].reshape([-1])
    zv = volume_gt['z'].reshape([-1])
    volume_vector = [xv,yv,zv]
    return volume, volume_vector