import os
import numpy as np
import torch
import scipy.io as scio
import h5py
import matplotlib.pyplot as plt
 

def load_zaragoza256_raw(basedir):
    nlos_data = h5py.File(basedir, 'r')
    # nlos_data = scio.loadmat(basedir)
    data  = nlos_data['data'][0,:,1,:,:]
    # data = np.array(nlos_data['data'])
    # data = data[:,1,:,:]
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

    M = camera_position.shape[1]
    N = camera_position.shape[2]

    dist1 = np.sqrt(np.sum((camera_grid_positions - camera_position.reshape([3,1,1])) ** 2, axis = 0))
    dist2 = np.sqrt(np.sum((camera_grid_positions - laser_position.reshape([3,1,1])) ** 2, axis = 0))
    bin1 = (dist1 / (c * deltaT)).reshape([1, M, N])
    bin2 = (dist2 / (c * deltaT)).reshape([1, M, N])

    for i in range(M):
        for j in range(N):
            data[:,i,j] = np.roll(data[:,i,j], - bin1 - bin2)

    plt.plot(np.sum(np.sum(data, axis = 2), axis = 1))
    plt.savefig('./testfig/sumhist.png')

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c