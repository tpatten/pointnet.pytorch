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

OBJECT_TRANSLATION_THRESHOLD = 0.01
HAND_VALID_THRESHOLD = 76.


class GraspLearner:
    def __init__(self, args):
        # Get the arguments
        self.hand_directory = args.hand
        self.object_directory = args.object

        # Load the regression model
        f_args = parse_model_filename(os.path.basename(args.pointnet_model))
        k_out = 9
        data_subset = f_args['data_subset']
        dropout_p = f_args['dropout_p']
        average_pool = f_args['average_pool']
        arch = f_args['arch']
        joint_set = f_args['joint_set']

        manualSeed = random.randint(1, 10000)  # fix seed
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # Set up the model
        self.regressor = load_regression_model(arch, k_out, dropout_p, average_pool, model=args.pointnet_model)
        if self.regressor is None:
            print('Unknown architecture specified')
            sys.exit(0)

        self.regressor.cuda()
        self.regressor = self.regressor.eval()

        # Load the point cloud of the object model
        self.obj_cloud_base = np.loadtxt(os.path.join(opt.object_model, 'points.xyz'))

        # Load the point cloud of the robot gripper
        self.gripper_cloud_base = np.loadtxt('hand_open_symmetric.xyz')

        # Get the demonstrated grasp
        self.learn_grasp()

    def learn_grasp(self):
        # For each file
        pose_history = []
        frame_id = 0
        while True:
            # Load the estimated hand and object poses
            frame_name = str(frame_id).zfill(4)
            hand_pose_filename = os.path.join(self.hand_directory, frame_name + '.pkl')
            object_pose_filename = os.path.join(self.object_directory, frame_name + '.pkl')
            # print('-- {}'.format(hand_pose_filename))

            # If both the hand and object are valid
            if os.path.exists(hand_pose_filename) and os.path.exists(object_pose_filename):
                if not pose_history:
                    # Load the hand and object poses
                    print('Start frame: {}'.format(frame_name))
                    pose_history.append(self.load_hand_object_poses(hand_pose_filename, object_pose_filename))
                else:
                    # Get the difference in pose of the object in this frame compared to the previous frame
                    hand_pts, hand_score, object_pose = self.load_hand_object_poses(hand_pose_filename, object_pose_filename)
                    translation_delta = np.linalg.norm(pose_history[0][2][:3, 3] - object_pose[:3, 3])
                    if translation_delta > OBJECT_TRANSLATION_THRESHOLD:
                        break
                    else:
                        pose_history.append((hand_pts, hand_score, object_pose))

            frame_id = frame_id + 1

        print('End frame: {}'.format(str(frame_id).zfill(4)))

        # Find the first valid hand pose
        frame_end_idx = len(pose_history) - 1
        frame_valid_idx = frame_end_idx
        for i, e in reversed(list(enumerate(pose_history))):
            if e[1] < HAND_VALID_THRESHOLD:
                frame_valid_idx = i
                break
        print('Valid hand frame: {}'.format(str(frame_valid_idx).zfill(4)))

        # Estimate the grasp at the last frame
        frame_end_idx_tf = self.predict_robot_grasp(pose_history[frame_end_idx][0])
        frame_valid_idx_tf = self.predict_robot_grasp(pose_history[frame_valid_idx][0])
        estimated_tf = np.eye(4)
        self.visualize_learned_grasp((frame_end_idx_tf, frame_valid_idx_tf, estimated_tf),
                                     pose_history[frame_valid_idx][2])


    @staticmethod
    def load_hand_object_poses(hand_pose_filename, object_pose_filename):
        # Get the joints of the hand
        hand_pkl = load_pickle_data(hand_pose_filename)
        hand_pts = hand_pkl['handJoints3D'].astype(np.float32)
        hand_score = hand_pkl['score']

        # Get the object transformation matrix
        object_pkl = load_pickle_data(object_pose_filename)
        obj_tf_mat = np.eye(4)
        obj_tf_mat[:3, :3] = cv2.Rodrigues(object_pkl['objRot'])[0].T
        obj_tf_mat[:3, 3] = object_pkl['objTrans']

        return hand_pts, hand_score, obj_tf_mat

    def predict_robot_grasp(self, hand_pts):
        # Prepare the input for the regression model
        point_set = np.copy(hand_pts)
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

        # Predict
        points_batch = points_batch.transpose(2, 1)
        points_batch = points_batch.cuda()
        pred = self.regressor(points_batch)
        pred_np = pred.data.cpu().numpy()[0]
        predicted_gripper_transform = error_def.to_transform(pred_np, offset, dist)

        return predicted_gripper_transform


    def visualize_learned_grasp(self, grasp_pose, object_transform):
        # Get the three grasp poses
        frame_end_idx_tf, frame_valid_idx_tf, estimated_tf = grasp_pose

        # Get the object point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(self.obj_cloud_base)
        pts = np.asarray(obj_pcd.points)
        pts = np.matmul(pts, np.linalg.inv(object_transform[:3, :3]))
        pts += object_transform[:3, 3]
        obj_pcd.points = o3d.utility.Vector3dVector(pts)

        # Load the gripper cloud
        gripper_pcd = o3d.geometry.PointCloud()
        gripper_pcd.points = o3d.utility.Vector3dVector(self.gripper_cloud_base)

        # End frame grasp
        gripper_pcd_end = o3d.geometry.PointCloud()
        pts = np.asarray(gripper_pcd.points)
        pts = np.matmul(pts, np.linalg.inv(frame_end_idx_tf[0:3, 0:3]))
        pts += frame_end_idx_tf[0:3, 3]
        gripper_pcd_end.points = o3d.utility.Vector3dVector(pts)

        # Latest valid grasp
        gripper_pcd_valid = o3d.geometry.PointCloud()
        pts = np.asarray(gripper_pcd.points)
        pts = np.matmul(pts, np.linalg.inv(frame_valid_idx_tf[0:3, 0:3]))
        pts += frame_valid_idx_tf[0:3, 3]
        gripper_pcd_valid.points = o3d.utility.Vector3dVector(pts)

        # Final estimated grasp
        gripper_pcd_estimated = o3d.geometry.PointCloud()
        pts = np.asarray(gripper_pcd.points)
        pts = np.matmul(pts, np.linalg.inv(estimated_tf[0:3, 0:3]))
        pts += estimated_tf[0:3, 3]
        gripper_pcd_estimated.points = o3d.utility.Vector3dVector(pts)

        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Object cloud
        obj_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(obj_pcd)

        # Final frame grasp pose
        gripper_pcd_end.paint_uniform_color([1., 0., 0.])
        vis.add_geometry(gripper_pcd_end)

        # Latest valid frame grasp pose
        gripper_pcd_valid.paint_uniform_color([0., 1., 0.])
        vis.add_geometry(gripper_pcd_valid)

        # Final estimated grasp pose
        gripper_pcd_estimated.paint_uniform_color([0., 0., 1.])
        vis.add_geometry(gripper_pcd_estimated)

        # End
        vis.run()
        vis.destroy_window()


def load_pickle_data(f_name):
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    return pickle_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--pointnet_model', type=str, required=True, help='path to the pointnet model to load')
    #parser.add_argument('--hand', type=str, required=True, help="path to the hand pose estimates")
    #parser.add_argument('--object', type=str, required=True, help="path to the object pose estimates")
    #parser.add_argument('--object_model', type=str, required=True,
    #                    help="path to the directory that stores the object model")

    opt = parser.parse_args()
    opt.pointnet_model = '/home/tpatten/Data/ICAS2020/Models/XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch17_newSplit_84.pth'
    opt.hand = '/home/tpatten/Data/Hands/HO3D/train/ABF10/hand_tracker/'
    opt.object = '/home/tpatten/Data/Hands/HO3D/train/ABF10/meta/'
    opt.object_model = '/home/tpatten/Data/Hands/HO3D/models/021_bleach_cleanser/'
    opt.verbose = True
    print(opt)

    learner = GraspLearner(opt)
