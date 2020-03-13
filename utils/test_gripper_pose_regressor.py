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
import open3d as o3d
import copy

num_points = 21

gripper_diameter = 0.232280153674483
add_threshold = 0.1 * gripper_diameter
adds_threshold = add_threshold
translation_threshold = 0.05
rotation_threshold = np.radians(5)


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
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")

    opt = parser.parse_args()
    opt.batchSize = 32
    opt.visualize = True
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
