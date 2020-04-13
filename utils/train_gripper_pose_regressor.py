from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from pointnet.dataset import HO3DDataset
from pointnet.model import PointNetRegression, compute_loss, MSE_LOSS_CODE, L1_LOSS_CODE, MODEL_LOSS_CODE
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
    '--batch_size', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument(
    '--model', type=str, default='', help='model path')
parser.add_argument(
    '--dataset', type=str, required=True, help="dataset path")
parser.add_argument(
    '--dropout_p', type=float, default=0.0, help='probability of dropout')
parser.add_argument(
    '--splitloss', action='store_true', help="compute loss separately for translation and directions")
parser.add_argument(
    '--closing_symmetry', action='store_true', help="consider loss function that is symmetric for the closing angle")
parser.add_argument(
    '--data_augmentation', action='store_true', help="apply data augmentation when training")
parser.add_argument(
    '--data_subset', type=str, default='ALL', help='subset of the dataset to train on')  # ALL ABF BB GPMF GSF MDF SHSU
parser.add_argument(
    '--randomly_flip_closing_angle', action='store_true', help="apply random flips to the closing angle")
parser.add_argument(
    '--center_to_wrist_joint', action='store_true', help="center the points using the wrist position")
parser.add_argument(
    '--model_loss', action='store_true', help="use the model loss (compute distance between points)")
parser.add_argument(
    '--average_pool', action='store_true', help="perform average pooling instead of max pooling")
parser.add_argument(
    '--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument(
    '--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument(
    '--learning_step', type=int, default=20, help='step size for learning rate scheduler')
parser.add_argument(
    '--learning_gamma', type=float, default=0.5, help='multiplier of the learning rate every step')
parser.add_argument(
    '--l1_loss', action='store_true', help="use l1 loss instead of mse (model_loss will override this)")
parser.add_argument(
    '--tensorboard', action='store_true', help="enable tensorboard")


opt = parser.parse_args()
opt.k_out = 9
opt.lc_weights = [1./3., 1./3., 1./3.]
opt.loss_reduction = 'mean'  # 'mean' or 'sum'
opt.save_model = True
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = HO3DDataset(root=opt.dataset, data_augmentation=opt.data_augmentation, subset_name=opt.data_subset,
                      randomly_flip_closing_angle=opt.randomly_flip_closing_angle,
                      center_to_wrist_joint=opt.center_to_wrist_joint)

test_dataset = HO3DDataset(
    root=opt.dataset,
    split='test',
    data_augmentation=False,
    subset_name=opt.data_subset,
    randomly_flip_closing_angle=opt.randomly_flip_closing_angle,
    center_to_wrist_joint=opt.center_to_wrist_joint)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers))

print('Size of train: {}\nSize of test: {}'. format(len(dataset), len(test_dataset)))

gripper_filename = 'hand_open_symmetric.xyz'
gripper_pts = np.loadtxt(gripper_filename)

loss_type = MSE_LOSS_CODE
output_dir = opt.data_subset + '_batch' + str(opt.batch_size) + '_nEpoch' + str(opt.nepoch)
if opt.model_loss:
    output_dir += '_modelLoss'
    loss_type = MODEL_LOSS_CODE
else:
    output_dir += '_regLoss'
if opt.dropout_p > 0.0:
    output_dir = output_dir + '_dropout' + str(opt.dropout_p).replace('.', '-')
if opt.data_augmentation:
    output_dir += '_augmentation'
if opt.splitloss:
    output_dir += '_splitLoss'
if opt.splitloss and opt.closing_symmetry:
    output_dir += '_symmetry'
elif opt.model_loss and opt.closing_symmetry:
    output_dir += '_symmetry'
if opt.randomly_flip_closing_angle:
    output_dir += '_rFlipClAng'
if opt.center_to_wrist_joint:
    output_dir += '_wristCentered'
if opt.average_pool:
    output_dir += '_avgPool'
if not opt.model_loss and opt.l1_loss:
    output_dir += '_l1Loss'
    loss_type = L1_LOSS_CODE
if opt.weight_decay > 0.0:
    output_dir = output_dir + '_wDecay' + str(opt.weight_decay).replace('.', '-')
output_dir = output_dir + '_lr' + str(opt.learning_rate).replace('.', '-') +\
             '_ls' + str(opt.learning_step) + '_lg' + str(opt.learning_gamma).replace('.', '-')
print('Output directory\n{}'.format(output_dir))

if opt.save_model:
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

regressor = PointNetRegression(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)

start_epoch = 0
if opt.model != '':
    regressor.load_state_dict(torch.load(opt.model))
    basename = os.path.basename(opt.model)
    filename, _ = os.path.splitext(basename)
    idx = filename.rfind('_')
    start_epoch = int(filename[idx + 1:]) + 1

optimizer = optim.Adam(regressor.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.learning_step, gamma=opt.learning_gamma)
regressor.cuda()

num_batch = len(dataset) / opt.batch_size

all_errors = {}
all_errors[error_def.ADD_CODE] = []

if opt.tensorboard:
    tensorboard_writer = SummaryWriter('/home/tpatten/logs/' + output_dir)

for epoch in range(start_epoch, opt.nepoch):
    epoch_loss = [0, 0]
    epoch_accuracy = [0, 0]
    num_tests = 0
    if epoch > start_epoch:
        scheduler.step()
    for i, data in enumerate(dataloader, 0):
        # Load the data
        points, target, offset, dist = data
        points = points.transpose(2, 1)

        # Put onto GPU
        points, target = points.cuda(), target.cuda()

        # Call train and predict
        optimizer.zero_grad()
        regressor = regressor.train()
        pred = regressor(points)

        # Compute loss value
        loss = compute_loss(pred, target, offset, dist, gripper_pts, loss_type=loss_type,
                            independent_components=opt.splitloss, lc_weights=opt.lc_weights,
                            closing_symmetry=opt.closing_symmetry, reduction=opt.loss_reduction)
        # if opt.model_loss:
        #     loss = model_loss(pred, target, offset, dist, gripper_pts, closing_symmetry=opt.closing_symmetry)
        # else:
        #     loss = mse_loss(pred, target, independent_components=opt.splitloss, lc_weights=opt.lc_weights,
        #                     closing_symmetry=opt.closing_symmetry, reduction=opt.loss_reduction)

        # Backprop
        loss.backward()
        optimizer.step()

        # Compute the accuracy
        targ_np = target.data.cpu().numpy()
        pred_np = pred.data.cpu().numpy()
        offset_np = offset.data.cpu().numpy().reshape((pred_np.shape[0], 3))
        dist_np = dist.data.cpu().numpy().reshape((pred_np.shape[0], 1))
        targ_tfs = error_def.to_transform_batch(targ_np, offset_np, dist_np)
        pred_tfs = error_def.to_transform_batch(pred_np, offset_np, dist_np)
        errs = error_def.get_all_errors_batch(targ_tfs, pred_tfs, gripper_pts, all_errors)
        evals = error_def.eval_grasps(errs, error_threshold, None, None)

        # Print to terminal
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), evals[0][0]))
        epoch_loss[0] += loss.item()
        epoch_accuracy[0] += evals[0][0]

        # Check accuracy on test set (validation)
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, offset, dist = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            regressor = regressor.eval()
            pred = regressor(points)
            loss_test = compute_loss(pred, target, offset, dist, gripper_pts, loss_type=loss_type,
                                     independent_components=opt.splitloss, lc_weights=opt.lc_weights,
                                     closing_symmetry=opt.closing_symmetry, reduction=opt.loss_reduction)
            # if opt.model_loss:
            #     loss_test = model_loss(pred, target, offset, dist, gripper_pts, closing_symmetry=opt.closing_symmetry)
            # else:
            #     if opt.l1_loss:
            #         loss_test = l1_loss(pred, target, independent_components=opt.splitloss, lc_weights=opt.lc_weights,
            #                             closing_symmetry=opt.closing_symmetry, reduction=opt.loss_reduction)
            #     else:
            #         loss_test = mse_loss(pred, target, independent_components=opt.splitloss, lc_weights=opt.lc_weights,
            #                              closing_symmetry=opt.closing_symmetry, reduction=opt.loss_reduction)
            targ_np = target.data.cpu().numpy()
            pred_np = pred.data.cpu().numpy()
            offset_np = offset.data.cpu().numpy().reshape((pred_np.shape[0], 3))
            dist_np = dist.data.cpu().numpy().reshape((pred_np.shape[0], 1))

            targ_tfs = error_def.to_transform_batch(targ_np, offset_np, dist_np)
            pred_tfs = error_def.to_transform_batch(pred_np, offset_np, dist_np)
            errs = error_def.get_all_errors_batch(targ_tfs, pred_tfs, gripper_pts, all_errors)
            evals_test = error_def.eval_grasps(errs, error_threshold, None, None)

            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'),
                                                            loss_test.item(), evals_test[0][0]))
            epoch_loss[1] += loss_test.item()
            epoch_accuracy[1] += evals_test[0][0]
            num_tests += 1

    # Save the checkpoint
    if opt.save_model and epoch != 0:
        torch.save(regressor.state_dict(), '%s/%s_%d.pth' % (output_dir, output_dir, epoch))

    # Only keep every 10th
    if opt.save_model and epoch > 0 and (epoch - 1) % 10 != 0:
        os.remove('%s/%s_%d.pth' % (output_dir, output_dir, epoch - 1))

    # To tensorboard
    if opt.tensorboard:
        epoch_loss[0] /= num_batch
        epoch_accuracy[0] /= num_batch
        epoch_loss[1] /= float(num_tests)
        epoch_accuracy[1] /= float(num_tests)
        tensorboard_writer.add_scalar('Loss/train', epoch_loss[0], epoch)
        tensorboard_writer.add_scalar('Loss/test', epoch_loss[1], epoch)
        tensorboard_writer.add_scalar('Accuracy/train', epoch_accuracy[0], epoch)
        tensorboard_writer.add_scalar('Accuracy/test', epoch_accuracy[1], epoch)
        #tensorboard_writer.add_scalars('Loss', {'train': epoch_loss[0], 'test': epoch_loss[1]}, epoch)
        #tensorboard_writer.add_scalars('Accuracy', {'train': epoch_accuracy[0], 'test': epoch_accuracy[1]}, epoch)
        #for tag, value in regressor.named_parameters():
        #    tag = tag.replace('.', '/')
        #    tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), epoch)
        #    if value.grad is not None:
        #        tensorboard_writer.add_histogram(tag + '/grad', value.grad.cpu().numpy(), epoch)

if opt.tensorboard:
    # Close the tensor board writer
    tensorboard_writer.close()

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

print('\n--- FINAL ERRORS ---')
print('ADD\tADDS\tT\tRx\tRy\tRz\tR')
print('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(np.mean(np.asarray(all_errors[error_def.ADD_CODE])),
                                            np.mean(np.asarray(all_errors[error_def.ADDS_CODE])),
                                            np.mean(np.asarray(all_errors[error_def.TRANSLATION_CODE])),
                                            np.mean(np.asarray(all_errors[error_def.ROTATION_X_CODE])),
                                            np.mean(np.asarray(all_errors[error_def.ROTATION_Y_CODE])),
                                            np.mean(np.asarray(all_errors[error_def.ROTATION_Z_CODE])),
                                            np.mean(np.asarray(all_errors[error_def.ROTATION_CODE]))))
print('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(np.std(np.asarray(all_errors[error_def.ADD_CODE])),
                                            np.std(np.asarray(all_errors[error_def.ADDS_CODE])),
                                            np.std(np.asarray(all_errors[error_def.TRANSLATION_CODE])),
                                            np.std(np.asarray(all_errors[error_def.ROTATION_X_CODE])),
                                            np.std(np.asarray(all_errors[error_def.ROTATION_Y_CODE])),
                                            np.std(np.asarray(all_errors[error_def.ROTATION_Z_CODE])),
                                            np.std(np.asarray(all_errors[error_def.ROTATION_CODE]))))

evals = error_def.eval_grasps(all_errors, add_threshold, adds_threshold, (translation_threshold, rotation_threshold))
print('\n--- FINAL CORRECT ({}) ---'.format(len(all_errors[error_def.ADDS_CODE])))
print('ADD\tADDS\tT/R\tT/Rxyz')
print('{}\t{}\t{}\t{}\n'.format(round(evals[0][1], 3), round(evals[1][1], 3),
                                round(evals[2][1], 3), round(evals[3][1], 3)))
print('--- FINAL ACCURACY ---')
print('ADD\tADDS\tT/R\tT/Rxyz')
print('{}\t{}\t{}\t{}\n'.format(round(evals[0][0], 3), round(evals[1][0], 3),
                                round(evals[2][0], 3), round(evals[3][0], 3)))
