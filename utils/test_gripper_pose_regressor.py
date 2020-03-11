from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import HO3DDataset
from pointnet.model import PointNetRegression
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import error_def

num_points = 21


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")

opt = parser.parse_args()
opt.batchSize = 32
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

test_dataset = HO3DDataset(
    root=opt.dataset,
    split='test',
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True)

regressor = PointNetRegression(k_out=9)
regressor.load_state_dict(torch.load(opt.model))
regressor.cuda()
regressor = regressor.eval()

error_thresholds = np.linspace(0.01, 0.1, num=25, endpoint=True)  # 0.01  # 10 mm (1 cm)
print(error_thresholds)

gripper_filename = 'hand_open.xyz'
gripper_pts = np.loadtxt(gripper_filename)

total_correct = np.zeros(error_thresholds.shape)
total_testset = np.zeros(error_thresholds.shape)
all_errors = [[], [], [], [], [], [], []]
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target, offset, dist = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred = regressor(points)
    targ_np = target.data.cpu().numpy()
    pred_np = pred.data.cpu().numpy()
    offset_np = offset.data.cpu().numpy().reshape((pred_np.shape[0], 3))
    dist_np = dist.data.cpu().numpy().reshape((pred_np.shape[0], 1))
    # errs = np.linalg.norm(targ_np - pred_np, axis=1)

    targ_tfs = error_def.to_transform_batch(targ_np, offset_np, dist_np)
    pred_tfs = error_def.to_transform_batch(pred_np, offset_np, dist_np)

    errs = error_def.get_all_errors_batch(targ_tfs, pred_tfs, gripper_pts)
    # print(i, i*opt.batchSize, (i+1)*opt.batchSize, (i+1)*opt.batchSize - i*opt.batchSize)
    # all_errors[i*opt.batchSize:i*opt.batchSize + errs.shape[0], :] = errs
    for j in range(7):
        all_errors[j].extend(errs[j])

    #for j in range(error_thresholds.shape[0]):
    #    correct = np.where(errs < error_thresholds[j])
    #    total_correct[j] += len(correct[0])
    #    total_testset[j] += points.size()[0]

#print("final accuracy")
#for j in range(error_thresholds.shape[0]):
#    # print("{}\t{}".format(error_thresholds[j], total_correct[j] / float(total_testset[j])))
#    print("{}".format(total_correct[j] / float(total_testset[j])))


for i in range(7):
    f = open(str(i) + '_results.txt', 'w')
    print('----- Metric {} -----'.format(i))
    for e in all_errors[i]:
        print(e)
        f.write("{}\n".format(e))
    f.close()
    print('')
