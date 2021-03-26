import os
import numpy as np
import torch
import scipy.io as scio
import h5py
import matplotlib.pyplot as plt
 

def load_zaragoza64_raw(basedir):
    nlos_data = h5py.File(basedir, 'r')
    # nlos_data = scio.loadmat(basedir)

    data = np.array(nlos_data['data'])
    data = data[:,1,:,:]
    # data = torch.from_numpy(data)



    deltaT = np.array(nlos_data['deltaT']).item()
    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    laser_position = np.array(nlos_data['laserPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.array(nlos_data['hiddenVolumeSize']).item()
    
    c = 1

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c