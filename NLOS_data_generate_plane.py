# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 17:49:35 2020

@author: zeromaker
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.io
from run_nerf_helpers import volume_box_point
from run_nerf_helpers import spherical2cartesian

def cartesian2spherical(pt):
    # 函数将直角坐标系下的点转换为球坐标系下的点
    # 输入格式： pt 是一个 N x 3 的 ndarray


    spherical_pt = np.zeros(pt.shape)
    spherical_pt[:,0] = np.sqrt(np.sum(pt ** 2,axis=1))
    spherical_pt[:,1] = np.arccos(pt[:,2] / spherical_pt[:,0])
    phi_yplus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] >= 0)
    phi_yplus = phi_yplus + (phi_yplus < 0).astype(np.int) * (np.pi)
    phi_yminus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] < 0)
    phi_yminus = phi_yminus + (phi_yminus > 0).astype(np.int) * (-np.pi)
    spherical_pt[:,2] = phi_yminus + phi_yplus

    # spherical_pt[:,2] = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) 
    # spherical_pt[:,2] = spherical_pt[:,2] + (spherical_pt[:,2] > 0).astype(np.int) * (-np.pi)

    return spherical_pt


# HDF5的读取：  
f = h5py.File('./data/bunny_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.21]_s[64]_l[64]_gs[0.60]_conf.mat','r')   #打开h5文件  
f.keys()                            #可以查看所有的主键 
 
# for key in f.keys():
#     print(f[key].name)
# print(f['deltaT'].value)
    
# data = f['data'][:]
cameraPosition = f['cameraPosition'][:]
cameraGridNormals = f['cameraGridNormals'][:]
cameraGridPositions = f['cameraGridPositions'][:]
cameraGridSize = f['cameraGridSize'][:]
cameraGridPoints = f['cameraGridPoints'][:].astype(np.int)
laserPosition = f['laserPosition'][:]
laserGridPositions = f['laserGridPositions'][:]
laserGridNormals = f['laserGridNormals'][:]
laserGridSize = f['laserGridSize'][:]
laserGridPoints = f['laserGridPoints'][:]
deltaT = f['deltaT'][()]
hiddenVolumePosition = f['hiddenVolumePosition'][:]
hiddenVolumeRotation = f['hiddenVolumeRotation'][:]
hiddenVolumeSize = f['hiddenVolumeSize'][()]
isConfocal = f['isConfocal'][()]
timeBins = f['t'][()]
initialTime = f['t0'][()]
f.close()


'''
nlos_data = scipy.io.loadmat('bunny_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.21]_s[64]_l[64]_gs[0.60]_conf.mat')
'''
# 参数初始化
hiddenVolumeSize[0,0] = 0.1 # 单位 m ， 指整个bounding box 是 0.2 x 0.2 x 0.2
hiddenVolumeResolution = 0.2 / 16 # 单位 m ， 指整个bounding box 是 0.2 x 0.2 x 0.2
M = int(hiddenVolumeSize[0,0] / hiddenVolumeResolution)
N = 3
L = M # M x N x L 是 object volume 的采样点数， N 是 Y 轴上的厚度，在墙这个例子中只有一层有值，其他都是0
c = 3e8
timeResolution = 4e-12
[m,n] = cameraGridPoints[0]
unit_length = c * timeResolution

'''
# # rho初始化
# rho = np.zeros([M,N,L])
# rho[:,0,:] = 1

# # volume坐标初始化, x~[-0.1,+0.1], y~[-0.5], z~[-0.1,+0.1]
# volumeLocation = np.zeros([3,M,N,L])
# for i in range(M):
#     volumeLocation[0,i,:,:] = i * hiddenVolumeResolution + hiddenVolumePosition[0] - hiddenVolumeResolution * M/2
# # a = volumeLocation[0,:,0,:]
# for j in range(N):
#     volumeLocation[1,:,j,:] = j * hiddenVolumeResolution + hiddenVolumePosition[1] - hiddenVolumeResolution * N/2
# # a = volumeLocation[1,:,0,:]
# for k in range(L):
#     volumeLocation[2,:,:,k] = k * hiddenVolumeResolution + hiddenVolumePosition[2] - hiddenVolumeResolution * L/2
# # a = volumeLocation[2,:,0,:]'''

box_point = volume_box_point(hiddenVolumePosition.reshape(-1), hiddenVolumeSize) # 返回物体box的八个顶点的直角坐标
hiddenVolumePosition_sph = np.array([0.5,np.pi/2,-np.pi/2])
sphere_box_point = cartesian2spherical(box_point)
theta_min = np.min(sphere_box_point[:,1])
theta_max = np.max(sphere_box_point[:,1])
thetaResolution = (theta_max - theta_min) / M
phi_min = np.min(sphere_box_point[:,2])
phi_max = np.max(sphere_box_point[:,2])
phiResolution = (phi_max - phi_min) / L
r_min = np.abs(hiddenVolumePosition[1]) - unit_length
r_max = np.abs(hiddenVolumePosition[1]) + unit_length
rResolution = (r_max - r_min) / N

# theta = np.linspace(theta_min, theta_max , 16)
# phi = np.linspace(phi_min, phi_max, 16)

# rho初始化(反射率)
rho_sph = np.zeros([N,M,L])
rho_sph[1,:,:] = 1

volumeLocation_sph = np.zeros([3,N,M,L])
for i in range(N):
    volumeLocation_sph[0,i,:,:] = i * rResolution + hiddenVolumePosition_sph[0] - rResolution * N/2
# a = volumeLocation[0,:,0,:]
for j in range(M):
    volumeLocation_sph[1,:,j,:] = j * thetaResolution + hiddenVolumePosition_sph[1] - thetaResolution * M/2
# a = volumeLocation[1,:,0,:]
for k in range(L):
    volumeLocation_sph[2,:,:,k] = k * phiResolution + hiddenVolumePosition_sph[2] - phiResolution * L/2
# a = volumeLocation[2,:,0,:]

volumeLocation = spherical2cartesian(volumeLocation_sph.reshape([3,M * N * L]).T).T

timeBins = timeBins[0][0]
# tau初始化
tau = np.zeros([m, n, timeBins])

# laser坐标初始化
laserPositions = laserGridPositions.reshape([3,m,n])

for i in range(0,m,1):
    for j in range(0,n,1):
        time0 = time.time()
        print(i,j)
        # 记k代表第k个时间间隔，[(k-1)*timeResolution,k*timeResolution]
        # 为了减少计算量，忽略前面部分点，最短距离是0.5m，则 416 是first returning photon
        p0 = laserPositions[:,i,j].reshape([3,-1])
        voLoSp = cartesian2spherical((volumeLocation.reshape([3,M * N * L]) + p0).T).T
        distance = voLoSp[0,:]
        y_value = volumeLocation.T[:,1]
        for k in range(400,700):
            distanceMin = ((k-1) * timeResolution * c )
            distanceMax = (k * timeResolution * c )
            indicator = np.ones(M * N * L) * (distance > distanceMin) * (distance < distanceMax) * (y_value > -0.5 -  10 * unit_length ) * (y_value < -0.5 + 10 * unit_length )
            tau[i,j,k] = np.sum(indicator) # / (distanceMax ** 2) 
            # if tau[i,j,700] > 0:
            #     error
            # if tau[i,j,400] > 0:
            #     error

            # for idx in range(M):
            #     for jdx in range(N):
            #         for kdx in range(L):
            #             a = volumeLocation[:,idx,jdx,kdx]
            #             b = laserPositions[:,i,j]
            #             distanceSquare = np.sum((a-b)**2)
            #             if ((distanceSquare > distanceSquareMin) & (distanceSquare < distanceSquareMax)):
            #                 tau[i,j,k] += 1

        dt = time.time()-time0
        print(i,'/',m,' ',j,'/',n,' time for this histogram is ',dt)

# taumax = np.max(tau)
# tau = tau / taumax

x = np.linspace(1,timeBins,timeBins) * timeResolution * c
y = tau[i,j,:] #/ np.max(tau[i,j,:])
plt.plot(x[:], y[:], color='red', marker='o', label='aaa')
plt.savefig('./datageneration/histogram_distance')
plt.close()
plt.plot(x[400:700], y[400:700], color='red', marker='o', label='aaa')
plt.savefig('./datageneration/part_histogram_distance')
plt.close()

x = np.linspace(1,timeBins,timeBins)
y = tau[i,j,:] #/ np.max(tau[i,j,:])
plt.plot(x[:], y[:], color='red', marker='o', label='aaa')
plt.savefig('./datageneration/histogram_timeBins')
plt.close()
plt.plot(x[400:700], y[400:700], color='red', marker='o', label='aaa')
plt.savefig('./datageneration/part_histogram_timeBins')
plt.close()


# # 绘制散点图
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(0,0,0,c='g')
# ax.scatter(volumeLocation[0,int(M/2),int(N/2),int(L/2)],volumeLocation[1,int(M/2),int(N/2),int(L/2)],volumeLocation[2,int(M/2),int(N/2),int(L/2)],c='g')
# ax.scatter(laserPositions[0,:,:].reshape([-1]), laserPositions[1,:,:].reshape([-1]), laserPositions[2,:,:].reshape([-1]), c = 'r')
# ax.scatter(volumeLocation[0,:,:,:].reshape([-1]), volumeLocation[1,:,:,:].reshape([-1]), volumeLocation[2,:,:,:].reshape([-1]), c = 'b', alpha = 0.1, linewidth = 0.1)

# # 添加坐标轴(顺序是Z, Y, X)
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# plt.savefig('./datageneration/scene')
# plt.close()

# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(0,0,0,c='g')
# ax.scatter(volumeLocation.reshape([3,M,N,L])[0,int(M/2),int(N/2),int(L/2)],volumeLocation[1,int(M/2),int(N/2),int(L/2)],volumeLocation[2,int(M/2),int(N/2),int(L/2)],c='g')
ax.scatter(laserPositions[0,:,:].reshape([-1]), laserPositions[1,:,:].reshape([-1]), laserPositions[2,:,:].reshape([-1]), c = 'r')
ax.scatter(volumeLocation[0,:], volumeLocation[1,:], volumeLocation[2,:], c = 'b', alpha = 0.1, linewidth = 0.1)

# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.savefig('./datageneration/scene')
plt.close()

mdic = {"data": tau, \
    "cameraPosition": cameraPosition, \
    "cameraGridSize": cameraGridSize, \
    "cameraGridPositions": cameraGridPositions, \
    "cameraGridPoints": cameraGridPoints,\
    "hiddenVolumePosition": hiddenVolumePosition,\
    "hiddenVolumeSize": hiddenVolumeSize }

scipy.io.savemat("./datageneration/NLOS_data_generated.mat", mdic)
    































