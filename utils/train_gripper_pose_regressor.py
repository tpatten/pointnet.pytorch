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

error_threshold = 0.1 * 0.232280153674483
add_threshold = error_threshold
adds_threshold = add_threshold
translation_threshold = 0.05
rotation_threshold = np.radians(5)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='gpr', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = HO3DDataset(root=opt.dataset)

test_dataset = HO3DDataset(
    root=opt.dataset,
    split='test',
    data_augmentation=False)


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))

gripper_filename = 'hand_open.xyz'
gripper_pts = np.loadtxt(gripper_filename)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

regressor = PointNetRegression(k_out=9)

start_epoch = 0
if opt.model != '':
    regressor.load_state_dict(torch.load(opt.model))
    basename = os.path.basename(opt.model)
    filename, _ = os.path.splitext(basename)
    idx = filename.rfind('_')
    start_epoch = int(filename[idx + 1:]) + 1

optimizer = optim.Adam(regressor.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
regressor.cuda()

num_batch = len(dataset) / opt.batchSize

all_errors = {}
all_errors[error_def.ADD_CODE] = []

for epoch in range(start_epoch, opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target, offset, dist = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        regressor = regressor.train()
        pred = regressor(points)
        loss = F.mse_loss(pred, target)  # Do I need to split each component? How about symmetry?
        loss.backward()
        optimizer.step()
        targ_np = target.data.cpu().numpy()
        pred_np = pred.data.cpu().numpy()
        offset_np = offset.data.cpu().numpy().reshape((pred_np.shape[0], 3))
        dist_np = dist.data.cpu().numpy().reshape((pred_np.shape[0], 1))

        targ_tfs = error_def.to_transform_batch(targ_np, offset_np, dist_np)
        pred_tfs = error_def.to_transform_batch(pred_np, offset_np, dist_np)
        errs = error_def.get_all_errors_batch(targ_tfs, pred_tfs, gripper_pts, all_errors)
        evals = error_def.eval_grasps(errs, error_threshold, None, None)

        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), evals[0]))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, offset, dist = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            regressor = regressor.eval()
            pred = regressor(points)
            loss = F.mse_loss(pred, target)
            targ_np = target.data.cpu().numpy()
            pred_np = pred.data.cpu().numpy()
            offset_np = offset.data.cpu().numpy().reshape((pred_np.shape[0], 3))
            dist_np = dist.data.cpu().numpy().reshape((pred_np.shape[0], 1))

            targ_tfs = error_def.to_transform_batch(targ_np, offset_np, dist_np)
            pred_tfs = error_def.to_transform_batch(pred_np, offset_np, dist_np)
            errs = error_def.get_all_errors_batch(targ_tfs, pred_tfs, gripper_pts, all_errors)
            evals = error_def.eval_grasps(errs, error_threshold, None, None)

            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), evals[0]))

    if epoch != 0:
        torch.save(regressor.state_dict(), '%s/gpr_model_%d.pth' % (opt.outf, epoch))

    # Only keep every 10th
    if epoch > 0 and (epoch - 1) % 10 != 0:
        os.remove('%s/gpr_model_%d.pth' % (opt.outf, epoch - 1))

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
    regressor = regressor.eval()
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

evals = error_def.eval_grasps(all_errors, add_threshold, adds_threshold, (translation_threshold, rotation_threshold))
print('\n--- FINAL ACCURACY ---')
print('ADD\tADDS\tT/R\tT/Rxyz')
print('{}\t{}\t{}\t{}'.format(round(evals[0], 3), round(evals[1], 3), round(evals[2], 3), round(evals[3], 3)))
print('')
