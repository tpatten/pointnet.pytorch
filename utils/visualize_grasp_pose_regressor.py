from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import HO3DDataset
from pointnet.model import PointNetRegression
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import pickle
import transforms3d as tf3d
import copy
import cv2
import error_def


def visualize(meta_filename, grasp_pose_filename, models_path):
    # Load the points and object model
    with open(meta_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    point_set = pickle_data['handJoints3D'].astype(np.float32)
    # center
    offset = np.expand_dims(np.mean(point_set, axis=0), 0)
    point_set -= offset
    # scale
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set /= dist
    # create tensor
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
    # gripper_pcd = o3d.io.read_point_cloud('hand_open_symmetric.ply')
    gripper_pcd_gt = copy.deepcopy(gripper_pcd)
    gripper_pcd_predicted = copy.deepcopy(gripper_pcd)

    # Compute the diameter
    '''
    centered_gripper_pts = copy.deepcopy(gripper_xyz)
    g_offset = np.expand_dims(np.mean(centered_gripper_pts, axis=0), 0)
    centered_gripper_pts = centered_gripper_pts - g_offset
    max_diameter = 2.0 * np.max(np.sqrt(np.sum(centered_gripper_pts ** 2, axis=1)), 0)
    print('Gripper diameter = {}'.format(max_diameter))
    '''

    # Transform with ground truth
    pts = np.asarray(gripper_pcd.points)
    pts = np.matmul(pts, np.linalg.inv(gripper_transform[:3, :3]))
    gripper_trans = gripper_transform[:3, 3]
    pts += gripper_trans
    gripper_pcd_gt.points = o3d.utility.Vector3dVector(pts)

    # Transform with the prediction
    gripper_trans_scaled = copy.deepcopy(pred_np[0:3])
    gripper_trans_scaled *= dist
    gripper_trans_scaled += offset.flatten()
    z_axis = copy.deepcopy(pred_np[3:6])
    z_axis /= np.linalg.norm(z_axis)
    y_axis = copy.deepcopy(pred_np[6:9])
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rot = np.eye(3)
    rot[:, 0] = x_axis
    rot[:, 1] = y_axis
    rot[:, 2] = z_axis
    pts = np.asarray(gripper_pcd.points)
    pts = np.matmul(pts, np.linalg.inv(rot))
    pts += gripper_trans_scaled
    gripper_pcd_predicted.points = o3d.utility.Vector3dVector(pts)
    predicted_gripper_transform = np.eye(4)
    predicted_gripper_transform[:3, :3] = rot
    predicted_gripper_transform[:3, 3] = gripper_trans_scaled

    print('GROUND TRUTH')
    print(gripper_transform)
    euler_gt = tf3d.euler.mat2euler(gripper_transform[:3, :3])
    print('Translation\n{}\n{}\n{}'.format(gripper_transform[0, 3], gripper_transform[1, 3], gripper_transform[2, 3]))
    print('Rotation\n{}\n{}\n{}'.format(np.degrees(euler_gt[0]), np.degrees(euler_gt[1]), np.degrees(euler_gt[2])))

    print('PREDICTION')
    print(pred_np)
    euler_prediction = tf3d.euler.mat2euler(rot)
    print('Translation\n{}\n{}\n{}'.format(gripper_trans_scaled[0], gripper_trans_scaled[1], gripper_trans_scaled[2]))
    print('Rotation\n{}\n{}\n{}'.format(np.degrees(euler_prediction[0]), np.degrees(euler_prediction[1]),
                                        np.degrees(euler_prediction[2])))

    print('ERROR')
    print('ADD {}'.format(error_def.add_error(gripper_transform, predicted_gripper_transform,
                                              np.asarray(gripper_pcd.points))))
    print('ADD Symmetric {}'.format(error_def.add_symmetric_error(gripper_transform, predicted_gripper_transform,
                                                                  np.asarray(gripper_pcd.points))))
    print('Translation {}'.format(error_def.translation_error(gripper_transform, predicted_gripper_transform)))
    print('Rotation {}'.format(np.degrees(error_def.rotation_error(gripper_transform, predicted_gripper_transform)[0])))

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
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
    # end
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")

    opt = parser.parse_args()
    opt.models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'
    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Set up the model
    regressor = PointNetRegression(k_out=9)
    regressor.load_state_dict(torch.load(opt.model))
    regressor.cuda()
    regressor = regressor.eval()

    # Get the files
    splitfile = os.path.join(opt.dataset, 'grasp_test.txt')
    f = open(splitfile, "r")
    filelist = [line[:-1] for line in f]
    f.close()

    # For each file
    for file in filelist:
        subject, seq = file.split('/')
        meta_filename = os.path.join(opt.dataset, 'train', subject, 'meta', seq + '.pkl')
        grasp_pose_filename = os.path.join(opt.dataset, 'train', subject, 'meta', 'grasp_bl_' + seq + '.pkl')

        print(' --- Processing file {} ---'.format(meta_filename))

        # Visualize
        visualize(meta_filename, grasp_pose_filename, opt.models_path)
        print('\n')
