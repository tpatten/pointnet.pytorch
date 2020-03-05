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

error_threshold = 0.01  # 10 mm (1 cm)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=21, help='input batch size')  # 2500
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
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
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points,
        regression=True)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False,
        regression=True)
else:
    exit('wrong dataset type')


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
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

regressor = PointNetRegression(k_out=3)

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

for epoch in range(start_epoch, opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        # target = target[:, 0]  # NEED something different (maybe)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        regressor = regressor.train()
        pred = regressor(points)
        # loss = F.nll_loss(pred, target)  # NEED different loss
        loss = F.mse_loss(pred, target)  # set reduction='mean' or 'sum'??
        loss.backward()
        optimizer.step()
        # mse_sum = F.mse_loss(pred, target, reduction='sum')
        # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(),
        #                                                    mse_sum.data.item() / float(opt.batchSize)))
        targ_np = target.data.cpu().numpy()
        pred_np = pred.data.cpu().numpy()
        errs = np.linalg.norm(targ_np - pred_np, axis=1)
        correct = np.where(errs < error_threshold)
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(),
                                                           len(correct[0]) / float(opt.batchSize)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            # target = target[:, 0]  # NEED something different (maybe)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            regressor = regressor.eval()
            pred = regressor(points)
            # loss = F.nll_loss(pred, target)  # NEED different loss
            loss = F.mse_loss(pred, target)  # set reduction='mean' or 'sum'??
            # pred_choice = pred.data.max(1)[1]  # NEED something different (maybe)
            # correct = pred_choice.eq(target.data).cpu().sum()  # NEED something different (maybe)
            # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
            # mse_sum = F.mse_loss(pred, target, reduction='sum')
            # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(),
            #                                                 mse_sum.data.item() / float(opt.batchSize)))
            targ_np = target.data.cpu().numpy()
            pred_np = pred.data.cpu().numpy()
            errs = np.linalg.norm(targ_np - pred_np, axis=1)
            correct = np.where(errs < error_threshold)
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(),
                                                            len(correct[0]) / float(opt.batchSize)))

    if epoch != 0:
        torch.save(regressor.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    # Only keep every 10th
    if (epoch - 1) % 10 != 0:
        os.remove('%s/cls_model_%d.pth' % (opt.outf, epoch - 1))

total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    # target = target[:, 0]  # NEED something different (maybe)
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    regressor = regressor.eval()
    pred = regressor(points)
    # pred_choice = pred.data.max(1)[1]  # NEED something different (maybe)
    # correct = pred_choice.eq(target.data).cpu().sum()  # NEED something different (maybe)
    # total_correct += correct.item()  # NEED something different (maybe)
    # total_testset += points.size()[0]
    # num_tests = points.size()[0]
    # mse_sum = F.mse_loss(pred, target, reduction='sum')
    # mse = mse_sum.data.item() / float(num_tests)
    targ_np = target.data.cpu().numpy()
    pred_np = pred.data.cpu().numpy()
    errs = np.linalg.norm(targ_np - pred_np, axis=1)
    correct = np.where(errs < error_threshold)
    total_correct += len(correct[0])
    total_testset += points.size()[0]  # num_tests

print("final accuracy {}".format(total_correct / float(total_testset)))