from __future__ import print_function
import argparse
import os
import sys
import random
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import HO3DDataset, JointSet
from pointnet.model import *
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import error_def
import pickle
import open3d as o3d
import copy

num_points = 21

gripper_diameter = 0.232280153674483
add_threshold = 0.1 * gripper_diameter
adds_threshold = add_threshold
translation_threshold = 0.05
rotation_threshold = np.radians(15)


def visualize(gt_pose, est_pose, pts):
    # Gripper point cloud
    gripper_pcd = o3d.geometry.PointCloud()
    gripper_pcd.points = o3d.utility.Vector3dVector(pts)
    gripper_pcd_gt = copy.deepcopy(gripper_pcd)
    gripper_pcd_est = copy.deepcopy(gripper_pcd)

    # Transform with ground truth
    pts = np.asarray(gripper_pcd.points)
    pts = np.matmul(pts, np.linalg.inv(gt_pose[:3, :3]))
    gripper_trans = gt_pose[:3, 3]
    pts += gripper_trans
    gripper_pcd_gt.points = o3d.utility.Vector3dVector(pts)

    # Transform with the prediction
    pts = np.asarray(gripper_pcd.points)
    pts = np.matmul(pts, np.linalg.inv(est_pose[:3, :3]))
    pts += est_pose[:3, 3]
    gripper_pcd_est.points = o3d.utility.Vector3dVector(pts)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # ground truth
    gripper_pcd_gt.paint_uniform_color([0., 1., 0.])
    vis.add_geometry(gripper_pcd_gt)
    # predicted
    gripper_pcd_est.paint_uniform_color([0., 0., 1.])
    vis.add_geometry(gripper_pcd_est)
    # end
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='', help='model path')
    parser.add_argument(
        '--dataset', type=str, required=True, help="dataset path")
    parser.add_argument(
        '--data_subset_test', type=str, default='', help='subset of the dataset to test') # ALL ABF BB GPMF GSF MDF SHSU
    parser.add_argument(
        '--visualize', action='store_true', help='visualize the predicted grasp pose')

    opt = parser.parse_args()
    f_args = parse_model_filename(os.path.basename(opt.model))
    opt.batch_size = f_args['batch_size']
    opt.lc_weights = [1. / 3., 1. / 3., 1. / 3.]
    opt.loss_reduction = 'mean'  # 'mean' or 'sum'
    opt.k_out = 9
    opt.data_augmentation = f_args['data_augmentation']
    opt.data_subset_train = f_args['data_subset']
    if opt.data_subset_test == '':
        opt.data_subset_test = f_args['data_subset']
    opt.dropout_p = f_args['dropout_p']
    opt.randomly_flip_closing_angle = f_args['randomly_flip_closing_angle']
    opt.center_to_wrist_joint = f_args['center_to_wrist_joint']
    opt.average_pool = f_args['average_pool']
    opt.arch = f_args['arch']
    opt.yaxis_norm = f_args['yaxis_norm']
    opt.joint_set = f_args['joint_set']
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    test_dataset = HO3DDataset(
        root=opt.dataset,
        split='test',
        data_augmentation=False,
        subset_name=opt.data_subset_test,
        randomly_flip_closing_angle=opt.randomly_flip_closing_angle,
        center_to_wrist_joint=opt.center_to_wrist_joint,
        selected_joints=opt.joint_set)

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True)

    regressor = None
    if opt.arch == Archs.PN:
        regressor = PointNetRegression(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_Sym:
        regressor = PointNetRegressionSym(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_FC4:
        regressor = PointNetRegressionFC4(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_FC4_Sym:
        regressor = PointNetRegressionFC4Sym(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_FC45:
        regressor = PointNetRegressionFC45(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_FC45_Sym:
        regressor = PointNetRegressionFC45Sym(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_Small_3L:
        regressor = PointNetRegressionSmall3Layers(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_Small_4L:
        regressor = PointNetRegressionSmall4Layers(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_Half:
        regressor = PointNetRegressionHalf(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_Half_FC4:
        regressor = PointNetRegressionHalfFC4(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_FC4_256:
        regressor = PointNetRegressionFC4_256(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_LReLu:
        regressor = PointNetRegressionLeakyReLu(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_Half_LReLu:
        regressor = PointNetRegressionHalfLeakyReLu(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_Flat:
        regressor = PointNetRegressionFlat(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_NoPool:
        regressor = PointNetRegressionNoPool(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_NoPoolSmall:
        regressor = PointNetRegressionNoPoolSmall(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    elif opt.arch == Archs.PN_Flat5Layer:
        regressor = PointNetRegressionFlat5Layer(k_out=opt.k_out, dropout_p=opt.dropout_p, avg_pool=opt.average_pool)
    else:
        print('Unknown architecture specified')
        sys.exit(0)

    regressor.load_state_dict(torch.load(opt.model))
    regressor.cuda()
    regressor = regressor.eval()

    gripper_filename = 'hand_open_symmetric.xyz'
    gripper_pts = np.loadtxt(gripper_filename)

    error_names = [error_def.ADD_CODE,
                   error_def.ADDS_CODE,
                   error_def.TRANSLATION_CODE,
                   error_def.ROTATION_CODE,
                   error_def.ROTATION_X_CODE,
                   error_def.ROTATION_Y_CODE,
                   error_def.ROTATION_Z_CODE]
    all_errors = {}
    for e in error_names:
        all_errors[e] = []

    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target, offset, dist = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred = regressor(points)
        if opt.yaxis_norm:
            pred = normalize_direction(pred)
            target = normalize_direction(target)
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
        print('\nCorrect:\t{}\t{}\t{}\t{}'.format(
            round(evals[0][1], 3), round(evals[1][1], 3), round(evals[2][1], 3), round(evals[3][1], 3)))
        print('\nAccuracy:\t{}\t{}\t{}\t{}'.format(
            round(evals[0][0], 3), round(evals[1][0], 3), round(evals[2][0], 3), round(evals[3][0], 3)))

        if opt.visualize:
            err_str = '       '
            for key in error_names:
                err_str = err_str + '\t' + key
            print(err_str)
            for j in range(len(targ_tfs)):
                err_str = 'Errors:'
                eval_str = 'Eval:  '
                tr_eval = error_def.eval_translation_rotation([errs[error_def.TRANSLATION_CODE][j]],
                                                              [errs[error_def.ROTATION_CODE][j]],
                                                              [errs[error_def.ROTATION_X_CODE][j]],
                                                              [errs[error_def.ROTATION_Y_CODE][j]],
                                                              [errs[error_def.ROTATION_Z_CODE][j]],
                                                              translation_threshold, rotation_threshold)
                for key in error_names:
                    if key == error_def.ROTATION_CODE or key == error_def.ROTATION_X_CODE or \
                            key == error_def.ROTATION_Y_CODE or key == error_def.ROTATION_Z_CODE:
                        err_str = err_str + '\t' + str(round(np.degrees(errs[key][j]), 2))
                        if key == error_def.TRANSLATION_CODE or key == error_def.ROTATION_Y_CODE or \
                                key == error_def.ROTATION_Z_CODE:
                            eval_str = eval_str + '\t ----'
                        elif key == error_def.ROTATION_CODE:
                            eval_str = eval_str + '\t' + str(int(tr_eval[0]))
                        elif key == error_def.ROTATION_X_CODE:
                            eval_str = eval_str + '\t' + str(int(tr_eval[1]))
                    else:
                        err_str = err_str + '\t' + str(round(errs[key][j], 3))
                        eval_str = eval_str + '\t' + str(
                            int(error_def.eval_add_adds([errs[key][j]], add_threshold)))

                print(err_str)
                print(eval_str)

                visualize(targ_tfs[j], pred_tfs[j], gripper_pts)

    print('\n--- FINAL ERRORS ---')
    print('ADD\tADDS\tT\tRx\tRy\tRz\tR')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
        np.mean(np.asarray(all_errors[error_def.ADD_CODE])),
        np.mean(np.asarray(all_errors[error_def.ADDS_CODE])),
        np.mean(np.asarray(all_errors[error_def.TRANSLATION_CODE])) * 1000,
        np.degrees(np.mean(np.asarray(all_errors[error_def.ROTATION_X_CODE]))),
        np.degrees(np.mean(np.asarray(all_errors[error_def.ROTATION_Y_CODE]))),
        np.degrees(np.mean(np.asarray(all_errors[error_def.ROTATION_Z_CODE]))),
        np.degrees(np.mean(np.asarray(all_errors[error_def.ROTATION_CODE])))))
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
        np.std(np.asarray(all_errors[error_def.ADD_CODE])),
        np.std(np.asarray(all_errors[error_def.ADDS_CODE])),
        np.std(np.asarray(all_errors[error_def.TRANSLATION_CODE])) * 1000,
        np.degrees(np.std(np.asarray(all_errors[error_def.ROTATION_X_CODE]))),
        np.degrees(np.std(np.asarray(all_errors[error_def.ROTATION_Y_CODE]))),
        np.degrees(np.std(np.asarray(all_errors[error_def.ROTATION_Z_CODE]))),
        np.degrees(np.std(np.asarray(all_errors[error_def.ROTATION_CODE])))))

    evals = error_def.eval_grasps(all_errors, add_threshold, adds_threshold,
                                  (translation_threshold, rotation_threshold))
    print('--- FINAL CORRECT ({}) ---'.format(len(all_errors[error_def.ADDS_CODE])))
    print('ADD\tADDS\tT/R\tT/Rxyz')
    print('{}\t{}\t{}\t{}\n'.format(evals[0][1], evals[1][1], evals[2][1], evals[3][1]))
    print('--- FINAL ACCURACY ---')
    print('ADD\tADDS\tT/R\tT/Rxyz')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(evals[0][0], evals[1][0], evals[2][0], evals[3][0]))

    save_filename = 'results/' + os.path.basename(opt.model).replace('.pth', '.pkl')
    with open(save_filename, 'wb') as f:
        pickle.dump(all_errors, f)

    save_filename = save_filename.replace('.pkl', '.txt')
    f = open(save_filename, 'w')
    for key in all_errors:
        f.write(str(key) + ' ')
        for e in all_errors[key]:
            f.write(str(e) + ' ')
        f.write('\n')
    f.close()
