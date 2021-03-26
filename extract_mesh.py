import torch
from load_nlos import load_nlos_data
from run_nerf_helpers import *
from scipy import io
import numpy as np

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("./data/epoch4.pt")
model.eval()

model.to(device)

nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_nlos_data("./data/NLOS_data_generated_Plane_confocal_v2.mat")

# res_vol = torch.zeros(64, 64, 1024).to(device)
# res_depth = torch.zeros(256, 256).to(device)

# for m in range(0, nlos_data.shape[1], 1):
#     for n in range(0, nlos_data.shape[2], 1):
#         # for j in range(nlos_data.shape[0]):
#         v_light = 3
        # x = camera_grid_positions[0, m]
        # z = camera_grid_positions[1, n]
        # y = - v_light * 4 * j  / 100

        # pt = torch.tensor([x, y, z], dtype=torch.float32).to(device)
        # network_res = model(pt)
v_light = 3 
# v_light = 3 * 1e8
# r = v_light * t * 1e-12
input_hist = [np.array([camera_grid_positions[0, m], - v_light * 4 * k  / 100 + camera_grid_positions[1, m], camera_grid_positions[2, m]]) for m in range(camera_grid_positions.shape[1]) for k in range(300, 800)]
input_hist = np.array(input_hist)
input_hist = encoding_sph(input_hist, 10)
input_hist = torch.Tensor(input_hist).to(device)
print(1)
network_res = model(input_hist)
print(2)
# network_res = network_res.reshape(64, 64, 1024)
# network_res = torch.sum(network_res, 1)

# res_vol[m, n, j] = network_res
# print(x, y, z)
# res_vol[m, n, :] = network_res[:, 0]
# print('Done')
# print(m, n)

# v_light = 3
# x = camera_grid_positions[0, m]
# z = camera_grid_positions[1, n]
# y = - v_light * 4 * j  / 100

# pt = torch.tensor([x, y, z], dtype=torch.float32).to(device)
# network_res = model(pt)

# input_hist = [np.array([camera_grid_positions[0, m], - v_light * 4 * k  / 100, camera_grid_positions[1, n]]) for k in range(nlos_data.shape[0])]
# input_hist = torch.Tensor(input_hist).to(device)
# network_res = model(input_hist)

# for m in range(512):
#     for n in range(512):
#         for j in range(nlos_data.shape[0]):
#             v_light = 3
#             x = -0.3 + 0.6 / 512 * m
#             y = -0.3 + 0.6 / 512 * n
#             z = v_light * 4 * j * 3 / 100

#             pt = torch.tensor([x, y, z], dtype=torch.float32).to(device)
#             network_res = model(pt)
            
#             res_vol[m, n, j] = network_res
            # print(x, y, z)
        # print(m, n)

network_res = network_res.to("cpu").detach().numpy()
io.savemat('res_vol.mat', {'res_vol': network_res})