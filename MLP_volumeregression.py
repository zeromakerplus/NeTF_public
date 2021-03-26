#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:12:45 2020

@author: zeromaker
"""
# -*- coding: utf-8 -*-
import torch
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from MLP import Network
from MLP import Network4
from run_nerf_helpers import encoding_batch_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



# def encoding(pt, L):
#     # coded_pt = torch.zeros(6 * L)
#     # logseq = torch.logspace(start=0, end=L-1, steps=L, base=2)
#     # xsin = torch.sin(logseq.mul(math.pi).mul(pt[0]))
#     # ysin = torch.sin(logseq.mul(math.pi).mul(pt[1]))
#     # zsin = torch.sin(logseq.mul(math.pi).mul(pt[2]))
#     # xcos = torch.cos(logseq.mul(math.pi).mul(pt[0]))
#     # ycos = torch.cos(logseq.mul(math.pi).mul(pt[1]))
#     # zcos = torch.cos(logseq.mul(math.pi).mul(pt[2]))
#     # coded_pt = torch.reshape(torch.cat((xsin,xcos,ysin,ycos,zsin,zcos)), (1, 6 * L))

#     return coded_pt



# used for test MLP
if __name__ == "__main__": 
    seed = 0
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

    datafile = './data/borndataset/bunny_volume_born_64x64.mat'
    datadic = scipy.io.loadmat(datafile)
    volume = datadic['Volume'].astype(np.float)
    xv = datadic['x'].reshape([-1])
    yv = datadic['y'].reshape([-1])
    zv = datadic['z'].reshape([-1])
    [M,L,N] = volume.shape
    
    encoding_dim = 5

    # Create input pixel coordinates in the unit square
    # coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    # x_test = np.stack(np.meshgrid(coords, coords), -1)
    # test_data = [x_test, img]
    # train_data = [x_test[::2,::2], img[::2,::2]]
    coords = np.stack(np.meshgrid(xv, yv, zv),-1) # coords
    coords = coords.transpose([1,0,2,3])
    coords = coords.reshape([-1,3])
    # normalization
    coords = (coords - np.min(coords, axis = 0)) / (np.max(coords, axis = 0) - np.min(coords, axis = 0))
    coords = encoding_batch_numpy(coords, encoding_dim)
    occupancy = volume.reshape([-1])

    '''skipstep = 1
    testcoords = coords[0::skipstep,:]
    testoccupancy = occupancy[0::skipstep]'''

    
    uprate = 1
    # MM = M * uprate
    # NN = N * uprate
    # LL = L * uprate
    MM = 64
    NN = 64
    LL = 64
    xv_up = np.linspace(xv[0],xv[-1], MM)
    yv_up = np.linspace(yv[0],yv[-1], NN)
    zv_up = np.linspace(zv[0],zv[-1], LL)
    testcoords = np.stack(np.meshgrid(xv_up, yv_up, zv_up),-1)
    testcoords = testcoords.transpose([1,0,2,3])
    testcoords = testcoords.reshape([-1,3])
    testcoords = (testcoords - np.min(testcoords, axis = 0)) / (np.max(testcoords, axis = 0) - np.min(testcoords, axis = 0))
    testcoords = encoding_batch_numpy(testcoords, encoding_dim)
    testoccupancy = np.zeros([MM * NN * LL])

    testcoords = torch.from_numpy(testcoords).float().to(device)

    '''
    uprate = 5
    MM = int(M * uprate * 1 / 8)
    LL = L * uprate
    NN = int(N * uprate * 1 / 8)

    xv_up = np.linspace(xv[27 - 4],xv[int(27 + 4)], MM)
    yv_up = np.linspace(yv[0],yv[-1], LL)
    zv_up = np.linspace(zv[23 - 4],zv[int(23 + 4)], NN)

    testcoords = np.stack(np.meshgrid(xv_up, yv_up, zv_up),-1)
    testcoords = testcoords.transpose([1,0,2,3])
    testcoords = testcoords.reshape([-1,3])
    # normalization
    pmin = np.array([-0.25,-0.75,-0.25])
    pmax = np.array([0.25,-0.25,0.25])
    testcoords = (testcoords - pmin) / (pmax - pmin)
    
    testcoords = encoding_batch_numpy(testcoords, encoding_dim)
    testoccupancy = np.zeros([MM * LL * NN])

    testcoords = torch.from_numpy(testcoords).float().to(device)'''

    # data1 = data1 * 2 - 1
    # data2 = data2 * 2 - 1

    # testdata1 = testdata1 * 2 - 1
    # testdata2 = testdata2 * 2 - 1

    model = Network4(6 * encoding_dim, 64, 1)
    model = model.to(device)
    # model = Network(3, 128, 1)
    # model_name = './volumeregressionmodel/test1_withPE/epoch' + str(180) + '_withoutPE.pt'
    # model = torch.load(model_name)
    batchsize = 128 * 128 * 16
    testbatchsize = 128 * 128 * 16

    learning_rate = 1e-4

    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    epoch = 10000

    errorseq = []
    for i in range(epoch):

        if i % 500 == 0:
            
            with torch.no_grad():
                test_output = torch.empty([MM * LL * NN])
                for j in range(int(MM * LL * NN / testbatchsize)):
                    test_input_batch = testcoords[0 + j * testbatchsize:testbatchsize + j * testbatchsize,:]
                    test_output_batch = model(test_input_batch).view(-1)
                    test_output[0 + j * testbatchsize:testbatchsize + j * testbatchsize] = test_output_batch
            # test_gt = torch.from_numpy(testoccupancy).float().to(device)
            # # error = torch.sum(torch.abs(test_output - test_gt)) / (M * N * L / skipstep / skipstep)
            # with torch.no_grad():
            #     error = criterion(test_output, test_gt)
            # errorseq.append(error.item())
            # # epoch 计算误差
            # print('epoch:',i,'/',epoch, ' test set error: ', error)
            # print('pause')
            # # for name,param in model.named_parameters():
            # #     print('层:',name,param.size())
            # #     print('权值梯度',param.grad)
            # #     print('权值',param)
            
            if i % 500 == 0:
                model_name = './volumeregressionmodel/epoch' + str(i) + '_withPE.pt'
                torch.save(model, model_name)
                print('save model in epoch ' + str(i))

                test_output = test_output.view(MM, LL, NN)
                test_output = test_output.numpy()
                mdic = {'volume':test_output, 'x':xv, 'y':yv, 'z':zv}
                scipy.io.savemat('./volumeregressionmodel/regression_volume_epoch' + str(i) + '.mat', mdic)
                print('save predicted volume in epoch ' + str(i))
                del test_output


        for j in range(int(M * N * L / batchsize)):
            inputbatch = torch.from_numpy(coords[0 + j * batchsize:batchsize + j * batchsize,:]).float().to(device)
            I_pred = model(inputbatch).to(device).view(-1)
            gt = torch.from_numpy(occupancy[0 + j * batchsize:batchsize + j * batchsize]).float().to(device)
            loss = criterion(I_pred, gt)
            if i % 1 == 0:
                print('epoch = ',i, ' batchth = ', j, ' loss = ', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # a = gt.numpy().reshape([M,L,N])
            # adic = {'volume':a}
            # scipy.io.savemat('./volumeregressionmodel/gt.mat', adic)
        

        if i % 500 == 499:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
                learning_rate = param_group['lr']
            print('learning rate is updated to ',learning_rate)

# # for inference with loaded model
# if __name__ == "__main__": 
#     seed = 0
#     torch.manual_seed(seed)            # 为CPU设置随机种子
#     torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
#     torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

#     datafile = './data/bunny_volume_thick.mat'
#     datadic = scipy.io.loadmat(datafile)
#     volume = datadic['Volume'].astype(np.float)

#     unit_length = 0.5 / 64
#     xv = datadic['x'].reshape([-1]) #+ #unit_length / 2
#     yv = datadic['y'].reshape([-1]) #+ #unit_length / 2
#     zv = datadic['z'].reshape([-1]) #+ #unit_length / 2
#     [M,L,N] = volume.shape
    
#     encoding_dim = 10

#     # Create input pixel coordinates in the unit square
#     # coords = np.linspace(0, 1, img.shape[0], endpoint=False)
#     # x_test = np.stack(np.meshgrid(coords, coords), -1)
#     # test_data = [x_test, img]
#     # train_data = [x_test[::2,::2], img[::2,::2]]
#     coords = np.stack(np.meshgrid(xv, yv, zv),-1) # coords
#     coords = coords.transpose([1,0,2,3])
#     coords = coords.reshape([-1,3])
#     # normalization
#     pmin = np.array([-0.25,-0.75,-0.25])
#     pmax = np.array([0.25,-0.25,0.25])
#     coords = (coords - pmin) / (pmax - pmin)
#     coords = encoding_batch_numpy(coords, encoding_dim)
#     occupancy = volume.reshape([-1])

#     skipstep = 1
#     testcoords = coords[0::skipstep,:]
#     testoccupancy = occupancy[0::skipstep]

    

#     # data1 = data1 * 2 - 1
#     # data2 = data2 * 2 - 1

#     # testdata1 = testdata1 * 2 - 1
#     # testdata2 = testdata2 * 2 - 1

#     model = Network4(6 * encoding_dim, 64, 1)
    
#     # model = Network(3, 128, 1)

#     model_name = './volumeregressionmodel/epoch' + str(300) + '_withPE.pt'
#     model = torch.load(model_name)

#     model = model.to(device)

#     batchsize = 64 * 64 * 64

#     learning_rate = 1e-4

#     # criterion = torch.nn.L1Loss()
#     criterion = torch.nn.MSELoss(reduction='mean')
#     # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#     epoch = 1000


#     test_input = torch.from_numpy(testcoords).float().to(device)
#     with torch.no_grad():
#         test_output = model(test_input).view(-1)
#     test_gt = torch.from_numpy(testoccupancy).float().to(device)
#     # error = torch.sum(torch.abs(test_output - test_gt)) / (M * N * L / skipstep / skipstep)
#     with torch.no_grad():
#         error = criterion(test_output, test_gt)
#     # epoch 计算误差
#     print('test set error: ', error)
#     print('pause')
#     # for name,param in model.named_parameters():
#     #     print('层:',name,param.size())
#     #     print('权值梯度',param.grad)
#     #     print('权值',param)
    

#     test_volume = test_output.view(M,L,N)
#     test_volume = test_volume.numpy()
#     mdic = {'volume':test_volume, 'x':xv, 'y':yv, 'z':zv}
#     scipy.io.savemat('./volumeregressionmodel/regression_volume.mat', mdic)
#     print('save predicted volume')




#     # a = gt.numpy().reshape([M,L,N])
#     # adic = {'volume':a}
#     # scipy.io.savemat('./volumeregressionmodel/gt.mat', adic)
        





