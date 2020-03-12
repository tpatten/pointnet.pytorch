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
import pickle

num_points = 21

gripper_diameter = 0.232280153674483
add_threshold = 0.1 * gripper_diameter
adds_threshold = add_threshold
translation_threshold = 0.05
rotation_threshold = np.radians(5)


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

gripper_filename = 'hand_open.xyz'
gripper_pts = np.loadtxt(gripper_filename)

all_errors = {}
all_errors[error_def.ADD_CODE] = []
all_errors[error_def.ADDS_CODE] = []
all_errors[error_def.TRANSLATION_CODE] = []
all_errors[error_def.ROTATION_CODE] = []
all_errors[error_def.ROTATION_X_CODE] = []
all_errors[error_def.ROTATION_Y_CODE] = []
all_errors[error_def.ROTATION_Z_CODE] = []

for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target, offset, dist = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred = regressor(points)
    targ_np = target.data.cpu().numpy()
    pred_np = pred.data.cpu().numpy()
    offset_np = offset.data.cpu().numpy().reshape((pred_np.shape[0], 3))
    dist_np = dist.data.cpu().numpy().reshape((pred_np.shape[0], 1))

    targ_tfs = error_def.to_transform_batch(targ_np, offset_np, dist_np)
    pred_tfs = error_def.to_transform_batch(pred_np, offset_np, dist_np)

    errs = error_def.get_all_errors_batch(targ_tfs, pred_tfs, gripper_pts, all_errors)
    for key in all_errors:
        all_errors[key].extend(errs[key])

    evals = error_def.eval_grasps(errs, add_threshold, adds_threshold, (translation_threshold, rotation_threshold))
    print('\nAccuracy:\t{}\t{}\t{}\t{}'.format(
        round(evals[0], 3), round(evals[1], 3), round(evals[2], 3), round(evals[3], 3)))

evals = error_def.eval_grasps(all_errors, add_threshold, adds_threshold, (translation_threshold, rotation_threshold))
print('\n--- FINAL ACCURACY ---')
print('ADD\tADDS\tT/R\tT/Rxyz')
print('{}\t{}\t{}\t{}'.format(round(evals[0], 3), round(evals[1], 3), round(evals[2], 3), round(evals[3], 3)))
print('')

save_filename = 'results/results.pkl'
with open(save_filename, 'wb') as f:
    pickle.dump(all_errors, f)

save_filename = save_filename.replace('pkl', 'txt')
f = open(save_filename, 'w')
for key in all_errors:
    f.write(str(key) + ' ')
    for e in all_errors[key]:
        f.write(str(e) + ' ')
    f.write('\n')
f.close()
