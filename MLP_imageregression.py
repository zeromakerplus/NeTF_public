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

def encoding(pt, L):
    coded_pt = torch.zeros(6 * L)
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2)
    xsin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,0].view(-1,1))
    xcos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,0].view(-1,1))
    ysin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,1].view(-1,1))
    ycos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,1].view(-1,1))
    coded_pt = torch.cat((xsin,xcos,ysin,ycos),axis = 1)
    return coded_pt

class Network(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network, self).__init__()

        # self.linear1 = torch.nn.Linear(D_in, H)
        # self.linear3 = torch.nn.Linear(H, H)
        # self.linear2 = torch.nn.Linear(H, D_out)

        self.linear = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(H, D_out),
        # torch.nn.Sigmoid()
        )

        # main = torch.nn.Sequential()
        # torch.nn.Sequential().add_module('linear0',torch.nn.Linear(D_in, H))
        # for t in range(num_layer - 1): # 
        #     main.add_module('linear{0}'.format(t + 1),torch.nn.Linear(H, H))
        #     main.add_module('relu{0}'.format(t + 1),torch.nn.ReLU(inplace=True))
        # main.add_module('linearlast',torch.nn.Linear(H, 1))


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # h_relu = self.linear1(x).clamp(min=0)
        # y_pred = self.linear2(h_relu)

        # y_pred = torch.nn.functional.relu(self.linear1(x))
        # for i in range(self.num_layer - 1):
        #     y_pred = torch.nn.functional.relu(self.linear3(y_pred))
        # y_pred = torch.nn.functional.relu(self.linear2(y_pred))

        # y_pred = self.linear(x)

        y_pred = self.linear(x)
        return y_pred

# used for test MLP
if __name__ == "__main__": 
    seed = 0
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

    datafile = './data/sparseimgplus.mat'
    datadic = scipy.io.loadmat(datafile)
    img = datadic['img']
    plt.imshow(img, cmap = 'gray')
    plt.savefig('./image_regression_model/originalimg.png')

    [M, N] = img.shape
    L = 10

    # Create input pixel coordinates in the unit square
    # coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    # x_test = np.stack(np.meshgrid(coords, coords), -1)
    # test_data = [x_test, img]
    # train_data = [x_test[::2,::2], img[::2,::2]]

    linseq = np.linspace(0, 2, img.shape[0], endpoint=False)
    coords = np.stack(np.meshgrid(linseq, linseq),-1) # coords
    data1 = torch.from_numpy(coords.reshape([-1,2])).float()
    data1 = encoding(data1, L)
    data2  = torch.from_numpy(img.reshape([-1])).float()

    testdata1 = data1
    testdata2 = data2
    

    model = Network(4 * L, 128, 1)
    batchsize = 512 * 512

    learning_rate = 1e-4

    criterion = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    epoch = 10000
    for i in range(epoch):
        # for t in range(N):
        #     x_t = x[t,:]
        #     y_t = y[t,:]
        #     # Forward pass: Compute predicted y by passing x to the model
        #     y_pred = model(x_t)
        #     # print(y_pred, x_t)
        #     # Compute and print loss
        #     loss = criterion(y_pred, y_t)
        #     # if t % 100 == 99:
        #     #     print(i, t, loss.item())
        
        #     # Zero gradients, perform a backward pass, and update the weights.
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        if i % 10 == 0:
            with torch.no_grad():
                y_pred_test = model(testdata1)
                error = torch.sum(torch.abs(y_pred_test.view(-1) - testdata2)) / (M * N)
            # epoch 计算误差
            print(i,'/',epoch,'/',int(M * N / batchsize), ' test set error: ', error)
            
            
            pred_img = y_pred_test.view(M, N).numpy()
            plt.imshow(pred_img, cmap = 'gray')
            plt.savefig('./image_regression_model/pred_img'+ str(i) + '.png')


        for j in range(int(M * N / batchsize)):
            I_pred = model(data1[0 + j * batchsize:batchsize + j * batchsize,:])
            loss = criterion(I_pred.view(-1), data2[0 + j * batchsize:batchsize + j * batchsize]) / batchsize
            if j % 100 == 0:
                print('epoch ',i,' / ',epoch, ' batchth ', j,' / ',int(M * N / batchsize), 'loss = ', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if i % 500 == 0:
            model_name = './image_regression_model/epoch' + str(i) + '_withPE.pt'
            torch.save(model, model_name)

        if i % 1000 == 999:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
                learning_rate = param_group['lr']
            print('learning rate is updated to ',learning_rate)

# if __name__ == "__main__": 
#     seed = 0
#     torch.manual_seed(seed)            # 为CPU设置随机种子
#     torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
#     torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

#     [M, N] = [64, 64]
#     batchsize = 1024
#     L = 10
#     i = 2500
#     model_name = './image_regression_model/epoch' + str(i) + '_withPE.pt'
#     model = torch.load(model_name)
#     criterion = torch.nn.MSELoss(reduction='sum')

#     datafile = './img2.mat'
#     datadic = scipy.io.loadmat(datafile)
#     img = datadic['img']
#     # plt.imshow(img, cmap = 'gray')
#     # plt.savefig('./2.png')

            
#     linseq = np.linspace(0, 1, M, endpoint=False)
#     coords = np.stack(np.meshgrid(linseq, linseq),-1) # coords
#     data1 = torch.from_numpy(coords.reshape([-1,2])).float()
#     data1 = encoding(data1, L)

#     # for j in range(int(M * N / batchsize)):
#     #     I_pred = model(data1[0 + j * batchsize:batchsize + j * batchsize,:])
#         # loss = criterion(I_pred.view(-1), data2[0 + j * batchsize:batchsize + j * batchsize]) / batchsize
#         # if i % 50 == 0:
#         #     print('epoch = ',i, ' batchth = ', j, ' loss = ', loss)
#     with torch.no_grad():
#         I_pred = model(data1)
    
#     pred_img = I_pred.view(M, N).numpy()
#     plt.imshow(pred_img, cmap = 'gray')
#     plt.savefig('./pred_img1.png')



