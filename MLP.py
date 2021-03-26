#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:12:45 2020

@author: zeromaker
"""
# -*- coding: utf-8 -*-
import torch
import time

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
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(H, H),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(H, H),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(H, H),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(H, H),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(H, H),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(H, H),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(H, H),
        torch.nn.LeakyReLU(0.1),
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

class Network4(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super(Network4, self).__init__()
        
        # self.linear = torch.nn.Sequential(
        # torch.nn.Linear(D_in, H),
        # torch.nn.LeakyReLU(0.1),
        # # torch.nn.Linear(H, H),
        # # torch.nn.LeakyReLU(0.1),
        # # torch.nn.Linear(H, H),
        # # torch.nn.LeakyReLU(0.1),
        # # torch.nn.Linear(H, H),
        # # torch.nn.LeakyReLU(0.1),
        # # torch.nn.Linear(H, H),
        # # torch.nn.LeakyReLU(0.1),
        # torch.nn.Linear(H, H),
        # torch.nn.LeakyReLU(0.1),
        # torch.nn.Linear(H, H),
        # torch.nn.LeakyReLU(0.1),
        # torch.nn.Linear(H, H),
        # torch.nn.LeakyReLU(0.1),
        # torch.nn.Linear(H, D_out),
        # torch.nn.Tanh()
        # # torch.nn.Sigmoid()
        # )

        self.linear = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(inplace = True),
        # torch.nn.Linear(H, H),
        # torch.nn.ReLU(inplace = True),
        # torch.nn.Linear(H, H),
        # torch.nn.ReLU(inplace = True),
        # torch.nn.Linear(H, H),
        # torch.nn.ReLU(inplace = True),
        # torch.nn.Linear(H, H),
        # torch.nn.ReLU(inplace = True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace = True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace = True),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(inplace = True),
        torch.nn.Linear(H, D_out),
        # torch.nn.Tanh()
        # torch.nn.Sigmoid()
        )

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

import torch
import time

class Network_S(torch.nn.Module):
    def __init__(self, D=8, H=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], no_rho=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network_S, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.no_rho = no_rho

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, H)] + [torch.nn.Linear(H, H) if i not in self.skips else torch.nn.Linear(H + input_ch, H) for i in range(D-1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + H, H//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if self.no_rho:
            self.output_linear = torch.nn.Linear(H, output_ch)
        else:
            self.feature_linear = torch.nn.Linear(H, H)
            self.alpha_linear = torch.nn.Linear(H, 1)
            self.rho_linear = torch.nn.Linear(H//2, 1)

        # self.linear = torch.nn.Sequential(
        #     torch.nn.Linear(D_in, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, D_out),
        #     # torch.nn.Sigmoid()
        # )


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # y_pred = self.linear(x)
        if self.no_rho:
            input_pts = x
            h = x
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.no_rho:
            outputs = self.output_linear(h)
        else:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = torch.nn.functional.relu(h)

            rho = self.rho_linear(h)
            outputs = torch.cat([rho, alpha], -1)

        # if self.use_viewdirs:
        #     alpha = self.alpha_linear(h)
        #     feature = self.feature_linear(h)
        #     h = torch.cat([feature, input_views], -1)

        #     for i, l in enumerate(self.views_linears):
        #         h = self.views_linears[i](h)
        #         h = F.relu(h)

        #     rgb = self.rgb_linear(h)
        #     outputs = torch.cat([rgb, alpha], -1)
        # else:
        #     outputs = self.output_linear(h)

        return outputs



class Network_S_Relu(torch.nn.Module):
    def __init__(self, D=8, H=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], no_rho=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network_S_Relu, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.no_rho = no_rho

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, H)] + [torch.nn.Linear(H, H) if i not in self.skips else torch.nn.Linear(H + input_ch, H) for i in range(D-1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + H, H//2)])
        # self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + H, H//2)] + [torch.nn.Linear(H//2, H//2) for i in range(7)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if self.no_rho:
            self.output_linear = torch.nn.Linear(H, output_ch)
        else:
            self.feature_linear = torch.nn.Linear(H, H)
            self.alpha_linear = torch.nn.Linear(H, 1)
            self.rho_linear = torch.nn.Linear(H//2, 1)

        # self.linear = torch.nn.Sequential(
        #     torch.nn.Linear(D_in, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(H, D_out),
        #     # torch.nn.Sigmoid()
        # )


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # y_pred = self.linear(x)
        if self.no_rho:
            input_pts = x
            h = x
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.no_rho:
            outputs = self.output_linear(h)
        else:
            alpha = self.alpha_linear(h)
            alpha = torch.abs(alpha)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = torch.nn.functional.relu(h)

            rho = self.rho_linear(h)
            rho = torch.abs(rho)
            outputs = torch.cat([rho, alpha], -1)

        # if self.use_viewdirs:
        #     alpha = self.alpha_linear(h)
        #     feature = self.feature_linear(h)
        #     h = torch.cat([feature, input_views], -1)

        #     for i, l in enumerate(self.views_linears):
        #         h = self.views_linears[i](h)
        #         h = F.relu(h)

        #     rgb = self.rgb_linear(h)
        #     outputs = torch.cat([rgb, alpha], -1)
        # else:
        #     outputs = self.output_linear(h)

        return outputs

# if __name__ == "__main__": # used for test MLP
#     seed = 0
#     torch.manual_seed(seed)            # 为CPU设置随机种子
#     torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
#     torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

#     N, D_in, H, D_out, Batchsize = 100, 3, 10, 2, 10


#     x = torch.rand(N, D_in)
#     y1 = (x[:,0] + 2 * x[:,1] + 3 * x[:,2]) / 6
#     y2 = (3 * x[:,0] + 2 * x[:,1] + x[:,2]) / 6
#     y = torch.stack([y1,y2], dim=1)
#     y_pred_train = torch.zeros(N, D_out)


#     N2 = 10
#     x_test = torch.rand(N2, D_in) 
#     y1 = (x_test[:,0] + 2 * x_test[:,1] + 3 * x_test[:,2]) / 6
#     y2 = (3 * x_test[:,0] + 2 * x_test[:,1] + x_test[:,2]) / 6
#     y_test = torch.stack([y1,y2], dim=1)
#     y_pred_test = torch.zeros(N2, D_out)

# N2 = 100
# x_test = torch.randn(N2, D_in)
# y1 = x_test[:,0] + 2 * x_test[:,1] + 3 * x_test[:,2]
# y2 = 3 * x_test[:,0] + 2 * x_test[:,1] + x_test[:,2]   
# y_test = torch.stack([y1,y2], dim=1)
# y_pred_test = torch.zeros(N2, D_out)

# Construct our model by instantiating the class defined above
# model = Network(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
# epoch = 1

# time0 = time.time()
# ret = model(x_test)

# for i in range(epoch):
    
    # for k in range(N2):
    #     x_te = x_test[k,:]
    #     y_te = y_test[k]
    #     y_pred_test[k,:] = model(x_te).view(1, D_out)
        
    # error = y_pred_test - y_test
    # lo = torch.sum(error ** 2)
    # print(i, lo)
    # for t in range(N2):
    #     x_t = x_test[t,:]
        # y_t = y_test[t]
        # Forward pass: Compute predicted y by passing x to the model
        # y_pred = model(x_t)
    
        # Compute and print loss
        # loss = criterion(y_pred, y_t)
        # if t % 100 == 99:
        #     print(i, t, loss.item())
    
        # Zero gradients, perform a backward pass, and update the weights.
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    # # Construct our model by instantiating the class defined above
    # model = Network(D_in, H, D_out)

    # learning_rate = 1e-4
    # print(list(model.children()))
    # print('Learning Rate is: ',learning_rate)

    # # Construct our loss function and an Optimizer. The call to model.parameters()
    # # in the SGD constructor will contain the learnable parameters of the two
    # # nn.Linear modules which are members of the model.
    # criterion = torch.nn.MSELoss(reduction='sum')
    # # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # epoch = 6000
    # for i in range(epoch):
        
    #     if i % 50 == 0:
    #         for k in range(N2):
    #             x_te = x_test[k,:]
    #             y_pred_test[k,:] = model(x_te).view(1, D_out)
    #         error = torch.abs(y_pred_test - y_test)
    #         lo_te = torch.sum(error) / (N2 * D_out)
    #         #  epoch 计算测试集误差

    #         for k in range(N):
    #             x_tr = x[k,:]
    #             y_pred_train[k,:] = model(x_tr).view(1, D_out)
    #         error = torch.abs(y_pred_train - y)
    #         lo_tr = torch.sum(error) / (N * D_out)
    #         #  epoch 计算测试集误差
    #         # print(y_pred_test)
    #         print(i, 'test set error: ', lo_te, '   training set error: ', lo_tr)

        
    #     # for t in range(N):
    #     #     x_t = x[t,:]
    #     #     y_t = y[t,:]
    #     #     # Forward pass: Compute predicted y by passing x to the model
    #     #     y_pred = model(x_t)
    #     #     # print(y_pred, x_t)
    #     #     # Compute and print loss
    #     #     loss = criterion(y_pred, y_t)
    #     #     # if t % 100 == 99:
    #     #     #     print(i, t, loss.item())
        
    #     #     # Zero gradients, perform a backward pass, and update the weights.
    #     #     optimizer.zero_grad()
    #     #     loss.backward()
    #     #     optimizer.step()

    #     y_pred = model(x)
    #     loss = criterion(y_pred, y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # print('training end')


    # print(y_test)
    # print(y_pred_test)


# dt = time.time()-time0
# print('Time: ', dt)


    





