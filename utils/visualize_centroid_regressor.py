from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetRegression
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument(
    '--file', type=str, required=True, help='input points')
parser.add_argument(
    '--num_points', type=int, default=21, help='input batch size')  # 2500
parser.add_argument('--model', type=str, default='', help='model path')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Set up the model
regressor = PointNetRegression()
regressor.load_state_dict(torch.load(opt.model))
regressor.cuda()
regressor = regressor.eval()

# Load the points
points_in = np.loadtxt(opt.file).astype(np.float32)
# resample
choice = np.random.choice(points_in.shape[0], opt.num_points, replace=True)
point_set = points_in[choice, :]
# center
offset = np.expand_dims(np.mean(point_set, axis=0), 0)
point_set = point_set - offset
# scale
dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
point_set = point_set / dist
# create tensor
points_batch = np.zeros((1, point_set.shape[0], point_set.shape[1]), dtype=np.float32)
points_batch[0, :, :] = point_set
points_batch = torch.from_numpy(points_batch)

# Predict
points_batch = points_batch.transpose(2, 1)
points_batch = points_batch.cuda()
pred = regressor(points_batch)
pred_np = pred.data.cpu().numpy()
# print(pred_np)

# Visualize
vis = o3d.visualization.Visualizer()
vis.create_window()
# original cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(points_in))
pcd.paint_uniform_color([0.2, 0.7, 0.2])
vis.add_geometry(pcd)
# sampled points
for i in range(point_set.shape[0]):
    mm = o3d.create_mesh_sphere(radius=0.025)
    mm.compute_vertex_normals()
    mm.paint_uniform_color([0.2, 0.2, 0.7])
    pos = point_set[i, :]
    pos *= dist
    pos += offset.flatten()
    tt = np.eye(4)
    tt[0:3, 3] = pos
    mm.transform(tt)
    vis.add_geometry(mm)
# centroid
mm = o3d.create_mesh_sphere(radius=0.04)
mm.compute_vertex_normals()
mm.paint_uniform_color([0.7, 0.2, 0.2])
tt = np.eye(4)
pred_np *= dist
pred_np += offset
tt[0:3, 3] = pred_np
mm.transform(tt)
vis.add_geometry(mm)
# coordinate frame
vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.5))
# end
vis.run()
vis.destroy_window()
