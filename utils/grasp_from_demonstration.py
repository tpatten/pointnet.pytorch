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
NUM_POINTS = 21
THRESHOLD = 0.5
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


class GraspLearner:
    def __init__(self, args):
        # Get the arguments
        self.dataset_directory = args.dataset
        self.target_name = args.target
        self.hand_directory = os.path.join(self.dataset_directory, self.target_name, 'hand_tracker')
        self.object_directory = os.path.join(self.dataset_directory, self.target_name, 'meta')
        self.rgb_directory = os.path.join(self.dataset_directory, self.target_name, 'rgb')
        self.depth_directory = os.path.join(self.dataset_directory, self.target_name, 'depth')

        # Load the regression model
        f_args = parse_model_filename(os.path.basename(args.pointnet_model))
        k_out = 9
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

        # Setup the hand pose estimator
        self.hand_estimator = HandPoseEstimator(args)

        # Load the point cloud of the object model
        self.obj_cloud_base = None
        if opt.object_model != '':
            self.obj_cloud_base = np.loadtxt(os.path.join(args.object_model, 'points.xyz'))

        # Load the point cloud of the robot gripper
        # self.gripper_cloud_base = np.loadtxt('hand_open_symmetric.xyz')
        self.gripper_cloud_base = np.loadtxt('hand_open.xyz')

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
        estimated_tf, hand_joints = self.estimate_final_grasp_pose(frame_valid_idx_tf, frame_valid_idx, frame_end_idx)
        object_frame_tf = np.matmul(np.linalg.inv(pose_history[frame_end_idx][2]), estimated_tf)
        self.visualize_learned_grasp((frame_valid_idx_tf, frame_end_idx_tf, estimated_tf, object_frame_tf),
                                     pose_history[frame_valid_idx][2], hand_joints)

    def load_hand_object_poses(self, hand_pose_filename, object_pose_filename):
        # Get the joints of the hand
        hand_pkl = load_pickle_data(hand_pose_filename)
        hand_pts = hand_pkl['handJoints3D'].astype(np.float32)
        hand_score = hand_pkl['score']

        # Get the object transformation matrix
        object_pkl = load_pickle_data(object_pose_filename)
        obj_tf_mat = np.eye(4)
        obj_tf_mat[:3, :3] = cv2.Rodrigues(object_pkl['objRot'])[0].T
        obj_tf_mat[:3, 3] = object_pkl['objTrans']

        if self.obj_cloud_base is None:
            self.obj_cloud_base = np.loadtxt(
                os.path.join('/home/tpatten/Data/Hands/HO3D/models', object_pkl['objName'], 'points.xyz'))

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

    def estimate_final_grasp_pose(self, grasp_pose, frame_valid, frame_end):
        rgb_val, depth_val = get_images(os.path.join(self.rgb_directory, str(frame_valid).zfill(4) + '.png'),
                                        os.path.join(self.depth_directory, str(frame_valid).zfill(4) + '.png'))
        pts_val = self.hand_estimator.process(rgb_val, depth_val)

        rgb_end, depth_end = get_images(os.path.join(self.rgb_directory, str(frame_end).zfill(4) + '.png'),
                                        os.path.join(self.depth_directory, str(frame_end).zfill(4) + '.png'))
        pts_end = self.hand_estimator.process(rgb_end, depth_end)

        final_grasp_pose = np.copy(grasp_pose)
        final_grasp_pose[:3, 3] += self.estimate_approach_vector(pts_val, pts_end)

        return final_grasp_pose, (pts_val, pts_end)

    @staticmethod
    def estimate_approach_vector(pts_val, pts_end):
        approach_vector = np.zeros((3, ))
        pair_count = 0
        for i in range(pts_val.shape[0]):
            if not np.isnan(pts_val[i][0]) and not np.isnan(pts_end[i][0]):
                approach_vector += pts_end[i][0] - pts_val[i][0]
                pair_count += 1

        return approach_vector / float(pair_count)

    def visualize_learned_grasp(self, grasp_pose, object_transform, hand_joints=None):
        # Get the three grasp poses
        frame_valid_idx_tf, frame_end_idx_tf, estimated_tf, object_frame_tf = grasp_pose

        # Get the object point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(self.obj_cloud_base)
        obj_pcd.transform(object_transform)

        # End frame grasp
        gripper_pcd_end = o3d.geometry.PointCloud()
        gripper_pcd_end.points = o3d.utility.Vector3dVector(self.gripper_cloud_base)
        gripper_pcd_end.transform(frame_end_idx_tf)

        # Latest valid grasp
        gripper_pcd_valid = o3d.geometry.PointCloud()
        gripper_pcd_valid.points = o3d.utility.Vector3dVector(self.gripper_cloud_base)
        gripper_pcd_valid.transform(frame_valid_idx_tf)

        # Final estimated grasp
        gripper_pcd_estimated = o3d.geometry.PointCloud()
        gripper_pcd_estimated.points = o3d.utility.Vector3dVector(self.gripper_cloud_base)
        gripper_pcd_estimated.transform(estimated_tf)

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

        # Hand joints
        if hand_joints is not None:
            pts_val, pts_end = hand_joints
            for i in range(len(all_indices)):
                for j in range(len(all_indices[i])):
                    if not np.isnan(pts_val[all_indices[i][j]][0]):
                        mm = o3d.create_mesh_sphere(radius=joint_sizes[j])
                        mm.compute_vertex_normals()
                        mm.paint_uniform_color(finger_colors[i])
                        trans3d = pts_val[all_indices[i][j]]
                        tt = np.eye(4)
                        tt[0:3, 3] = trans3d
                        mm.transform(tt)
                        vis.add_geometry(mm)
                    if not np.isnan(pts_end[all_indices[i][j]][0]):
                        mm = o3d.create_mesh_box(width=joint_sizes[j], height=joint_sizes[j], depth=joint_sizes[j])
                        mm.compute_vertex_normals()
                        mm.paint_uniform_color(finger_colors[i])
                        trans3d = pts_end[all_indices[i][j]]
                        tt = np.eye(4)
                        tt[0:3, 3] = trans3d
                        mm.transform(tt)
                        vis.add_geometry(mm)

        # Visualize the grasp in the object's frame
        obj_pcd_origin = o3d.geometry.PointCloud()
        obj_pcd_origin.points = o3d.utility.Vector3dVector(self.obj_cloud_base)
        obj_pcd_origin.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(obj_pcd_origin)

        object_frame_grasp_pcd = o3d.geometry.PointCloud()
        object_frame_grasp_pcd.points = o3d.utility.Vector3dVector(self.gripper_cloud_base)
        object_frame_grasp_pcd.transform(object_frame_tf)
        object_frame_grasp_pcd.paint_uniform_color([1., 0., 0.])
        vis.add_geometry(object_frame_grasp_pcd)

        # Visualize some random grasps
        for i in range(5):
            # Random object pose
            random_tf = np.eye(4)
            random_tf[:3, :3] = tf3d.euler.euler2mat(np.random.uniform(-np.pi, np.pi),
                                                     np.random.uniform(-np.pi, np.pi),
                                                     np.random.uniform(-np.pi, np.pi))
            random_tf[:3, 3] = np.random.uniform(-1, 1, (3,))

            # Object cloud
            obj_pcd_rand = o3d.geometry.PointCloud()
            obj_pcd_rand.points = o3d.utility.Vector3dVector(self.obj_cloud_base)
            obj_pcd_rand.transform(random_tf)
            obj_pcd_rand.paint_uniform_color([0.7, 0.7, 0.7])
            vis.add_geometry(obj_pcd_rand)

            # Gripper cloud
            gripper_pcd_rand = o3d.geometry.PointCloud()
            gripper_pcd_rand.points = o3d.utility.Vector3dVector(np.copy(self.gripper_cloud_base))
            #gripper_pcd_rand.transform(object_frame_tf)
            #gripper_pcd_rand.transform(random_tf)
            grasp_tf = np.matmul(random_tf, object_frame_tf)
            gripper_pcd_rand.transform(grasp_tf)
            if i == 0:
                gripper_pcd_rand.paint_uniform_color([0.7, 0.3, 0.3])
            elif i == 1:
                gripper_pcd_rand.paint_uniform_color([0.3, 0.7, 0.3])
            elif i == 2:
                gripper_pcd_rand.paint_uniform_color([0.3, 0.3, 0.7])
            elif i == 3:
                gripper_pcd_rand.paint_uniform_color([0.7, 0.7, 0.3])
            elif i == 4:
                gripper_pcd_rand.paint_uniform_color([0.7, 0.3, 0.7])
            vis.add_geometry(gripper_pcd_rand)

        # End
        vis.run()
        vis.destroy_window()


class HandPoseEstimator:
    def __init__(self, args):
        self.proto_file = args.proto_file
        self.weights_file = args.weights_file
        self.width = 640
        self.height = 480
        self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)

    def process(self, rgb_image, depth_image):
        self.width = rgb_image.shape[1]
        self.height = rgb_image.shape[0]
        aspect_ratio = self.width / self.height

        # Input image dimensions for the network
        in_height = 368
        in_width = int(((aspect_ratio * in_height) * 8) // 8)
        net_input = cv2.dnn.blobFromImage(rgb_image, 1.0 / 255, (in_width, in_height), (0, 0, 0),
                                          swapRB=False, crop=False)

        self.net.setInput(net_input)

        pred = self.net.forward()

        points = self.get_keypoints(pred)

        points3D = self.get_world_coordinates(points, depth_image)
        coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        points3D = points3D.dot(coord_change_mat.T)
        points3D = to_iccv_format(points3D)

        return points3D

    def get_keypoints(self, pred):
        # Empty list to store the detected keypoints
        points = []

        for i in range(NUM_POINTS):
            # confidence map of corresponding body's part.
            prob_map = pred[0, i, :, :]
            prob_map = cv2.resize(prob_map, (self.width, self.height))

            # Find global maxima of the probability map
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            if prob > THRESHOLD:
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        return points

    @staticmethod
    def get_world_coordinates(points, depth_image):
        points_3D = np.zeros((len(points), 3))

        ux = 312.42
        uy = 241.42
        fx = 617.343
        fy = 617.343
        i_fx = 1 / fx
        i_fy = 1 / fy

        for i in range(len(points)):
            if points[i] is not None:
                points_3D[i, 2] = depth_image[points[i][1], points[i][0]]
                points_3D[i, 0] = (points[i][0] - ux) * points_3D[i, 2] * i_fx
                points_3D[i, 1] = (points[i][1] - uy) * points_3D[i, 2] * i_fy
            else:
                points_3D[i, :] = np.nan

        return points_3D


def load_pickle_data(f_name):
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    return pickle_data


def get_images(rgb_filename, dep_filename):
    rgb_image = cv2.imread(rgb_filename)
    dep_image = cv2.imread(dep_filename)
    depth_scale = 0.00012498664727900177
    dep_image = dep_image[:, :, 2] + dep_image[:, :, 1] * 256
    dep_image = dep_image * depth_scale

    return rgb_image, dep_image


def to_iccv_format(joints):
    # MONOHAND [Wrist (0),
    #           TMCP (1), TPIP (2), TDIP (3), TTIP (4),
    #           IMCP (5), IPIP (6), IDIP (7), ITIP (8),
    #           MMCP (9), MPIP (10), MDIP (11), MTIP (12),
    #           RMCP (13), RPIP (14), RDIP (15), RTIP (16),
    #           PMCP (17), PPIP (18), PDIP (19), PTIP (20)]

    # ICCV     [Wrist,
    #           TMCP, IMCP, MMCP, RMCP, PMCP,
    #           TPIP, TDIP, TTIP,
    #           IPIP, IDIP, ITIP,
    #           MPIP, MDIP, MTIP,
    #           RPIP, RDIP, RTIP,
    #           PPIP, PDIP, PTIP]

    # joint_map = [0, 1, 5, 9, 13, 17, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

    # HO3D     [Wrist,
    #           IMCP, IPIP, IDIP,
    #           MMCP, MPIP, MDIP,
    #           PMCP, PPIP, PDIP
    #           RMCP, RPIP, RDIP,
    #           TMCP, TPIP, TDIP,
    #           TTIP, ITIP, MTIP, RTIP, PTIP]

    joint_map = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]

    iccv_joints = np.zeros(joints.shape)
    for i in range(len(joints)):
        iccv_joints[i, :] = joints[joint_map[i], :]

    return iccv_joints


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--pointnet_model', type=str, required=True, help='path to the pointnet model to load')
    #parser.add_argument('--hand', type=str, required=True, help="path to the hand pose estimates")
    #parser.add_argument('--object', type=str, required=True, help="path to the object pose estimates")
    #parser.add_argument('--object_model', type=str, required=True,
    #                    help="path to the directory that stores the object model")

    opt = parser.parse_args()
    opt.target = 'ABF10'
    opt.pointnet_model = '/home/tpatten/Data/ICAS2020/Models/XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch17_newSplit_84.pth'
    opt.dataset = '/home/tpatten/Data/Hands/HO3D/train'
    opt.object_model = ''  # '/home/tpatten/Data/Hands/HO3D/models/021_bleach_cleanser/'
    opt.proto_file = '/home/tpatten/Code/hand-tracking/caffe_models/pose_deploy.prototxt'
    opt.weights_file = '/home/tpatten/Code/hand-tracking/caffe_models/pose_iter_102000.caffemodel'
    opt.verbose = True
    print(opt)

    learner = GraspLearner(opt)
