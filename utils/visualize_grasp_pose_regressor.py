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
import numpy as np
import open3d as o3d
import pickle
import transforms3d as tf3d
import copy
import cv2
import error_def


wrist_indices = [0]
thumb_indices = [13, 14, 15, 16]
index_indices = [1, 2, 3, 17]
middle_indices = [4, 5, 6, 18]
ring_indices = [10, 11, 12, 19]
pinky_indices = [7, 8, 9, 20]
all_indices = [wrist_indices, thumb_indices, index_indices, middle_indices, ring_indices, pinky_indices]
finger_colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
finger_colors_box = [[0.2, 0.2, 0.2], [0.8, 0.2, 0.8], [0.2, 0.2, 0.8], [0.2, 0.8, 0.2], [0.8, 0.8, 0.2], [0.8, 0.2, 0.2]]
joint_sizes = [0.008, 0.006, 0.004, 0.004]
PASSGREEN = lambda x: '\033[92m' + x + '\033[0m'
FAILRED = lambda x: '\033[91m' + x + '\033[0m'
DIAMETER = 0.232280153674483


def load_files(dataset_path, data_subset):
    # Get the files
    subset_name = data_subset
    if subset_name[0] == 'X':
        subset_name = subset_name[1:]
    # splitfile = os.path.join(opt.dataset, 'grasp_test.txt')
    filename = os.path.join(dataset_path, 'splits_new', subset_name + '_grasp_test.txt')
    f = open(filename, "r")
    files = [line[:-1] for line in f]
    f.close()

    return files


def visualize(meta_filename, hand_filename, grasp_pose_filename, models_path, verbose=False):
    # Load the points
    with open(hand_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    point_set = pickle_data['handJoints3D'].astype(np.float32)
    joint_to_mean_dists = np.sqrt(np.sum((point_set - np.nanmean(point_set, axis=0)) ** 2, axis=1))
    invalid_joints = np.isnan(np.sum(point_set, axis=1))
    num_invalid_joints = np.count_nonzero(invalid_joints)
    point_set = np.nan_to_num(point_set)
    joint_to_mean_dists = np.nan_to_num(joint_to_mean_dists)

    outlier_joints = joint_to_mean_dists > 0.5
    invalid_joints = np.logical_or(invalid_joints, outlier_joints)
    for i in range(len(invalid_joints)):
        if np.all((point_set[i] == 0)):
            invalid_joints[i] = True

    # Load the object information
    with open(meta_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    # Replace the missing values
    gt_point_set = None
    if not meta_filename == hand_filename:
        if verbose:
            print('Missing values for {} joints'.format(num_invalid_joints))
        gt_point_set = pickle_data['handJoints3D'].astype(np.float32)
        for i in range(len(invalid_joints)):
            if invalid_joints[i]:
                point_set[i] = gt_point_set[i]
    point_set_copy = np.copy(point_set)

    # Center
    offset = np.expand_dims(np.mean(point_set, axis=0), 0)
    point_set -= offset
    # Scale
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set /= dist

    # Create tensor
    points_batch = np.zeros((1, point_set.shape[0], point_set.shape[1]), dtype=np.float32)
    points_batch[0, :, :] = point_set
    points_batch = torch.from_numpy(points_batch)

    # Get the object point cloud
    obj_rot = pickle_data['objRot']
    obj_trans = pickle_data['objTrans']
    obj_id = pickle_data['objName']
    obj_cloud_filename = os.path.join(models_path, obj_id, 'points.xyz')
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np.loadtxt(obj_cloud_filename))
    pts = np.asarray(obj_pcd.points)
    pts = np.matmul(pts, cv2.Rodrigues(obj_rot)[0].T)
    pts += obj_trans
    obj_pcd.points = o3d.utility.Vector3dVector(pts)

    # Predict
    points_batch = points_batch.transpose(2, 1)
    points_batch = points_batch.cuda()
    pred = regressor(points_batch)
    pred_np = pred.data.cpu().numpy()[0]

    # Load the ground truth grasp pose
    with open(grasp_pose_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    gripper_transform = pickle_data.reshape(4, 4)

    # Load the gripper cloud
    gripper_pcd = o3d.geometry.PointCloud()
    gripper_pcd.points = o3d.utility.Vector3dVector(np.loadtxt('hand_open_symmetric.xyz'))
    gripper_pcd_gt = copy.deepcopy(gripper_pcd)
    gripper_pcd_predicted = copy.deepcopy(gripper_pcd)

    # Transform with ground truth
    pts = np.asarray(gripper_pcd.points)
    pts = np.matmul(pts, np.linalg.inv(gripper_transform[:3, :3]))
    gripper_trans = gripper_transform[:3, 3]
    pts += gripper_trans
    gripper_pcd_gt.points = o3d.utility.Vector3dVector(pts)

    # Transform with the prediction
    predicted_gripper_transform = error_def.to_transform(pred_np, offset, dist)
    gripper_trans_scaled = copy.deepcopy(predicted_gripper_transform[0:3, 3])
    rot = copy.deepcopy(predicted_gripper_transform[0:3, 0:3])
    pts = np.asarray(gripper_pcd.points)
    pts = np.matmul(pts, np.linalg.inv(rot))
    pts += gripper_trans_scaled
    gripper_pcd_predicted.points = o3d.utility.Vector3dVector(pts)

    if verbose:
        print('GROUND TRUTH')
        print(gripper_transform)
        euler_gt = tf3d.euler.mat2euler(gripper_transform[:3, :3])
        print('Translation\n{}\n{}\n{}'.format(gripper_transform[0, 3], gripper_transform[1, 3],
                                               gripper_transform[2, 3]))
        print('Rotation\n{}\n{}\n{}'.format(np.degrees(euler_gt[0]), np.degrees(euler_gt[1]), np.degrees(euler_gt[2])))

        print('PREDICTION')
        print(pred_np)
        euler_prediction = tf3d.euler.mat2euler(rot)
        print('Translation\n{}\n{}\n{}'.format(gripper_trans_scaled[0], gripper_trans_scaled[1],
                                               gripper_trans_scaled[2]))
        print('Rotation\n{}\n{}\n{}'.format(np.degrees(euler_prediction[0]), np.degrees(euler_prediction[1]),
                                            np.degrees(euler_prediction[2])))

    print('ERROR')
    print('ADD {:.4f}'.format(error_def.add_error(gripper_transform, predicted_gripper_transform,
                                                  np.asarray(gripper_pcd.points))))
    adds_error = error_def.add_symmetric_error(gripper_transform, predicted_gripper_transform,
                                               np.asarray(gripper_pcd.points))
    print('ADD Symmetric {:.4f}'.format(adds_error))
    print('Translation {:.4f}'.format(error_def.translation_error(gripper_transform, predicted_gripper_transform)*100))
    rot_error = np.degrees(error_def.rotation_error(gripper_transform, predicted_gripper_transform))
    print('Rotation {:.2f}'.format(rot_error[0]))
    print('Rx {:.2f} Ry {:.2f} Rz {:.2f}'.format(rot_error[1], rot_error[2], rot_error[3]))
    if adds_error < 0.1 * DIAMETER:
        print('10%%: %s' % (PASSGREEN('PASS')))
    else:
        print('10%%: %s' % (FAILRED('FAIL')))
    if adds_error < 0.2 * DIAMETER:
        print('20%%: %s' % (PASSGREEN('PASS')))
    else:
        print('20%%: %s' % (FAILRED('FAIL')))

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Coordinate frame
    vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))
    # original cloud
    # gripper_pcd.paint_uniform_color([0.4, 0.4, 0.4])
    # vis.add_geometry(gripper_pcd)
    # ground truth
    gripper_pcd_gt.paint_uniform_color([0., 1., 0.])
    vis.add_geometry(gripper_pcd_gt)
    # predicted
    gripper_pcd_predicted.paint_uniform_color([0., 0., 1.])
    vis.add_geometry(gripper_pcd_predicted)
    # Object cloud
    obj_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(obj_pcd)
    # Ground truth joints
    if gt_point_set is not None:
        for i in range(len(all_indices)):
            for j in range(len(all_indices[i])):
                mm = o3d.create_mesh_sphere(radius=joint_sizes[j])
                mm.compute_vertex_normals()
                mm.paint_uniform_color(finger_colors[i])
                trans3d = gt_point_set[all_indices[i][j]]
                tt = np.eye(4)
                tt[0:3, 3] = trans3d
                mm.transform(tt)
                vis.add_geometry(mm)
    # Estimated joints
    for i in range(len(all_indices)):
        for j in range(len(all_indices[i])):
            if not invalid_joints[all_indices[i][j]]:
                mm = o3d.create_mesh_box(width=joint_sizes[j], height=joint_sizes[j], depth=joint_sizes[j])
                mm.compute_vertex_normals()
                mm.paint_uniform_color(finger_colors[i])
            else:
                mm = o3d.create_mesh_box(width=joint_sizes[0], height=joint_sizes[0], depth=joint_sizes[0])
                mm.compute_vertex_normals()
                mm.paint_uniform_color([0.5, 0.5, 0.5])
            trans3d = point_set_copy[all_indices[i][j]]
            tt = np.eye(4)
            tt[0:3, 3] = trans3d
            mm.transform(tt)
            vis.add_geometry(mm)

    # End
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--est', action='store_true', help="use the estimated hand joints")

    opt = parser.parse_args()
    opt.models_path = os.path.join(opt.dataset, 'models')
    opt.verbose = True
    print(opt)

    f_args = parse_model_filename(os.path.basename(opt.model))
    opt.k_out = 9
    opt.data_subset = f_args['data_subset']
    opt.dropout_p = f_args['dropout_p']
    opt.average_pool = f_args['average_pool']
    opt.arch = f_args['arch']
    opt.joint_set = f_args['joint_set']
    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Set up the model
    regressor = load_regression_model(opt.arch, opt.k_out, opt.dropout_p, opt.average_pool, model=opt.model)
    if regressor is None:
        print('Unknown architecture specified')
        sys.exit(0)

    regressor.cuda()
    regressor = regressor.eval()

    # Get the files
    filelist = load_files(opt.dataset, opt.data_subset)

    # For each file
    for file in filelist:
        subject, seq = file.split('/')
        meta_filename = os.path.join(opt.dataset, 'train', subject, 'meta', seq + '.pkl')
        if opt.est:
            hand_filename = os.path.join(opt.dataset, 'train', subject, 'hand_tracker', seq + '.pkl')
        else:
            hand_filename = os.path.join(opt.dataset, 'train', subject, 'meta', seq + '.pkl')
        if os.path.exists(hand_filename):
            grasp_pose_filename = os.path.join(opt.dataset, 'train', subject, 'meta', 'grasp_bl_' + seq + '.pkl')
            print('\n--- Processing file {} ---'.format(hand_filename))
            # Visualize
            visualize(meta_filename, hand_filename, grasp_pose_filename, opt.models_path, opt.verbose)
            sys.exit(0)
        else:
            print('No hand estimate for {}'.format(hand_filename))
