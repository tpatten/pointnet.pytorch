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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=21, help='input batch size')  # 2500
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False,
        regression=True)
else:
    exit('wrong dataset type')

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True)

regressor = PointNetRegression()
regressor.load_state_dict(torch.load(opt.model))
regressor.cuda()
regressor = regressor.eval()

error_thresholds = np.linspace(0.01, 0.1, num=25, endpoint=True)  # 0.01  # 10 mm (1 cm)
print(error_thresholds)
sys.exit(0)

total_correct = np.zeros(error_thresholds.shape)
total_testset = np.zeros(error_thresholds.shape)
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    # regressor = regressor.eval()
    pred = regressor(points)
    targ_np = target.data.cpu().numpy()
    pred_np = pred.data.cpu().numpy()
    errs = np.linalg.norm(targ_np - pred_np, axis=1)

    for j in range(error_thresholds.shape[0]):
        correct = np.where(errs < error_thresholds[j])
        total_correct[j] += len(correct[0])
        total_testset[j] += points.size()[0]

print("final accuracy")
for j in range(error_thresholds.shape[0]):
    print("{}\t{}".format(error_thresholds[j], total_correct[j] / float(total_testset[j])))