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

OBJECT_TRANSLATION_THRESHOLD = 0.1
HAND_VALID_THRESHOLD = 76.
NUM_POINTS = 21
THRESHOLD = 0.5
MAX_FRAMES = 100
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
        self.pix2pose = args.pix2pose
        self.hand_offset = args.hand_offset
        if self.pix2pose:
            self.hand_directory = os.path.join(self.dataset_directory, 'hand_tracker')
            self.object_directory = os.path.join(self.dataset_directory, 'object_pose')
            self.rgb_directory = os.path.join(self.dataset_directory, 'rgb')
            self.depth_directory = os.path.join(self.dataset_directory, 'depth')
        else:
            self.hand_directory = os.path.join(self.dataset_directory, self.target_name, 'hand_tracker')
            self.object_directory = os.path.join(self.dataset_directory, self.target_name, 'meta')
            self.rgb_directory = os.path.join(self.dataset_directory, self.target_name, 'rgb')
            self.depth_directory = os.path.join(self.dataset_directory, self.target_name, 'depth')
        self.ground_truth_prefix = args.ground_truth_prefix
        self.start_frame = args.start_frame
        self.save = args.save
        self.combine_grasp_files = args.combine_grasp_files

        if self.combine_grasp_files:
            self.combine_grasps()
            return

        # Load the regression model
        f_args = parse_model_filename(os.path.basename(args.pointnet_model))
        k_out = 9
        dropout_p = f_args['dropout_p']
        average_pool = f_args['average_pool']
        arch = f_args['arch']
        #joint_set = f_args['joint_set']

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
        frame_id = self.start_frame
        while True:
            # Load the estimated hand and object poses
            frame_name = str(frame_id).zfill(4)
            hand_pose_filename = os.path.join(self.hand_directory, frame_name + '.pkl')
            object_pose_filename = os.path.join(self.object_directory, frame_name + '.pkl')

            if len(pose_history) > 0 and not os.path.exists(object_pose_filename):
                print('No pose available, object is occluded and grasp can be computed from previous frame')
                break

            # If both the hand and object are valid
            if os.path.exists(hand_pose_filename) and os.path.exists(object_pose_filename):
                print('-- {}'.format(hand_pose_filename))
                # Get the hand points, score and object pose
                hand_pts, hand_score, object_pose = self.load_hand_object_poses(hand_pose_filename,
                                                                                object_pose_filename)
                # If the hand is valid
                if not np.isnan(hand_pts).all():
                    # If no pose has been recorded yet
                    if not pose_history:
                        print('Start frame: {}'.format(frame_name))
                        pose_history.append((hand_pts, hand_score, object_pose, frame_id))
                    # Otherwise, get the difference in pose of the object in this frame compared to the first frame
                    else:
                        translation_delta = np.linalg.norm(pose_history[0][2][:3, 3] - object_pose[:3, 3])
                        print('{}: {}'.format(frame_id, translation_delta))
                        if translation_delta > OBJECT_TRANSLATION_THRESHOLD:
                            print('Detected object movement {} at frame {}'.format(translation_delta, frame_id))
                            break
                        else:
                            pose_history.append((hand_pts, hand_score, object_pose, frame_id))
                    #if self.pix2pose:
                    #    self.visualize_scene(frame_id, object_pose, hand_pts)
            else:
                if not os.path.exists(hand_pose_filename):
                    print('{} does not exit'.format(hand_pose_filename))
                if not os.path.exists(object_pose_filename):
                    print('{} does not exit'.format(object_pose_filename))

            frame_id = frame_id + 1
            if frame_id >= self.start_frame + MAX_FRAMES:
                print('Object did not move within {} frames'.format(MAX_FRAMES))
                if len(pose_history) < 2:
                    print('No pose estimates available in the window')
                else:
                    print('Maximum distance is {:2f} cm'.format(translation_delta * 100))
                return

        print('End frame: {}'.format(str(frame_id).zfill(4)))

        # Find the first valid hand pose
        frame_end_idx = len(pose_history) - 1
        frame_valid_idx = frame_end_idx
        for i, e in reversed(list(enumerate(pose_history))):
            print(e[1])
            if e[1] < HAND_VALID_THRESHOLD:
                frame_valid_idx = i
                break
        print('Valid hand frame: {}'.format(str(pose_history[frame_valid_idx][-1]).zfill(4)))

        # Estimate the grasp at the end and valid frames
        frame_end_idx_tf = self.predict_robot_grasp(pose_history[frame_end_idx][0])
        frame_valid_idx_tf = self.predict_robot_grasp(pose_history[frame_valid_idx][0])

        # Estimate the final grasp using the approach vector
        estimated_tf, hand_joints = self.estimate_final_grasp_pose(
            frame_valid_idx_tf, pose_history[frame_valid_idx][0], pose_history[frame_valid_idx][-1],
            pose_history[frame_end_idx][-1])
        #if self.pix2pose:
        #    hand_joints = (pose_history[frame_valid_idx][0], pose_history[frame_valid_idx][0])

        # Transform to the object coordinate system
        object_frame_tf = np.matmul(np.linalg.inv(pose_history[frame_end_idx][2]), estimated_tf)

        # Get the ground truth grasp pose
        if self.ground_truth_prefix != '':
            gt_filename = os.path.join(self.dataset_directory, self.target_name,
                self.ground_truth_prefix + str(pose_history[frame_end_idx][-1]).zfill(4) + '.pkl')
            ground_truth_tf = load_pickle_data(gt_filename).reshape(4, 4)
            # Calculate the error
            self.compute_error(ground_truth_tf, estimated_tf)
        else:
            ground_truth_tf = None

        # Save
        if self.save:
            self.save_to_file(object_frame_tf)

        # Visualize the results
        self.visualize_learned_grasp(
            (frame_valid_idx_tf, frame_end_idx_tf, estimated_tf, object_frame_tf, ground_truth_tf),
            pose_history[frame_valid_idx][2], hand_joints)

    def load_hand_object_poses(self, hand_pose_filename, object_pose_filename):
        # Get the joints of the hand
        hand_pkl = load_pickle_data(hand_pose_filename)
        hand_pts = hand_pkl['handJoints3D'].astype(np.float32)
        hand_pts[:, 2] = hand_pts[:, 2] + self.hand_offset
        hand_score = hand_pkl['score']

        # Get the object transformation matrix
        object_pkl = load_pickle_data(object_pose_filename)
        obj_tf_mat = np.eye(4)
        rot = object_pkl['objRot']
        trans = object_pkl['objTrans']
        if rot.shape == (3, 3):
            obj_tf_mat[:3, :3] = rot
        else:
            obj_tf_mat[:3, :3] = cv2.Rodrigues(rot)[0].T
        obj_tf_mat[:3, 3] = trans

        if self.obj_cloud_base is None:
            if self.pix2pose:
                self.obj_cloud_base = np.asarray(o3d.read_point_cloud(
                    os.path.join('/home/tpatten/Data/ycbv/models', object_pkl['objName'] + '.ply')).points) * 0.001
            else:
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

    def estimate_final_grasp_pose(self, grasp_pose, hand_pts, frame_valid, frame_end):
        rgb_val, depth_val = get_images(os.path.join(self.rgb_directory, str(frame_valid).zfill(4) + '.png'),
                                        os.path.join(self.depth_directory, str(frame_valid).zfill(4) + '.png'),
                                        self.pix2pose)
        pts_val = self.hand_estimator.process(rgb_val, depth_val)

        rgb_end, depth_end = get_images(os.path.join(self.rgb_directory, str(frame_end).zfill(4) + '.png'),
                                        os.path.join(self.depth_directory, str(frame_end).zfill(4) + '.png'),
                                        self.pix2pose)
        pts_end = self.hand_estimator.process(rgb_end, depth_end)

        approach_vec = self.estimate_approach_vector(pts_val, pts_end)

        final_grasp_pose = np.copy(grasp_pose)
        final_grasp_pose[:3, 3] += approach_vec

        final_hand_pts = np.copy(hand_pts)
        final_hand_pts += approach_vec

        return final_grasp_pose, (hand_pts, final_hand_pts)

    @staticmethod
    def estimate_approach_vector(pts_val, pts_end):
        approach_vector = np.zeros((3, ))
        #return approach_vector
        pair_count = 0
        for i in range(pts_val.shape[0]):
            if not np.isnan(pts_val[i][0]) and not np.isnan(pts_end[i][0]):
                approach_vector += pts_end[i][0] - pts_val[i][0]
                pair_count += 1
                
        if pair_count == 0:
            print('No available pairs between frames')
            return approach_vector

        print('Approach vector\n', approach_vector / float(pair_count))

        return approach_vector / float(pair_count)

    def compute_error(self, gt_tf, est_tf):
        add_error = error_def.add_error(gt_tf, est_tf, self.gripper_cloud_base)
        adds_error = error_def.add_symmetric_error(gt_tf, est_tf, self.gripper_cloud_base)
        t_error = error_def.translation_error(gt_tf, est_tf)
        r_error = error_def.rotation_error(gt_tf, est_tf)

        print('ERROR')
        print('ADD {:.4f}'.format(add_error))
        print('ADD Symmetric {:.4f}'.format(adds_error))
        print('Translation {:.4f} cm'.format(t_error * 100))
        print('Rotation {:.2f} deg'.format(np.degrees(r_error[0])))
        print('Rx {:.2f} Ry {:.2f} Rz {:.2f}'.format(
            np.degrees(r_error[1]), np.degrees(r_error[2]), np.degrees(r_error[3])))

        return add_error, adds_error, t_error, r_error

    def visualize_learned_grasp(self, grasp_pose, object_transform, hand_joints=None):
        # Get the three grasp poses
        frame_valid_idx_tf, frame_end_idx_tf, estimated_tf, object_frame_tf, ground_truth_tf = grasp_pose

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

        # Ground truth grasp
        if ground_truth_tf is not None:
            gripper_pcd_gt = o3d.geometry.PointCloud()
            gripper_pcd_gt.points = o3d.utility.Vector3dVector(self.gripper_cloud_base)
            gripper_pcd_gt.transform(ground_truth_tf)

        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Object cloud
        obj_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(obj_pcd)

        # Final frame grasp pose
        #gripper_pcd_end.paint_uniform_color([0., 0., 1.])
        #vis.add_geometry(gripper_pcd_end)

        # Latest valid frame grasp pose
        #gripper_pcd_valid.paint_uniform_color([0., 1., 1.])
        #vis.add_geometry(gripper_pcd_valid)

        # Final estimated grasp pose
        gripper_pcd_estimated.paint_uniform_color([1., 0., 0.])
        vis.add_geometry(gripper_pcd_estimated)

        # Ground truth grasp pose
        if ground_truth_tf is not None:
            gripper_pcd_gt.paint_uniform_color([0., 1., 0.])
            vis.add_geometry(gripper_pcd_gt)

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

        '''
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
        '''

        '''
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
        '''

        # End
        vis.run()
        vis.destroy_window()

    def visualize_scene(self, frame_id, object_transform, hand_joints=None):
        # Get the object point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(self.obj_cloud_base)
        obj_pcd.transform(object_transform)
        #coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        #obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(obj_pcd.points).dot(coord_change_mat.T))

        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Object cloud
        obj_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(obj_pcd)

        # Hand joints
        if hand_joints is not None:
            for i in range(len(all_indices)):
                for j in range(len(all_indices[i])):
                    if not np.isnan(hand_joints[all_indices[i][j]][0]):
                        mm = o3d.create_mesh_sphere(radius=joint_sizes[j])
                        mm.compute_vertex_normals()
                        mm.paint_uniform_color(finger_colors[i])
                        trans3d = hand_joints[all_indices[i][j]]
                        tt = np.eye(4)
                        tt[0:3, 3] = trans3d
                        mm.transform(tt)
                        vis.add_geometry(mm)

        ux = 315.3074696331638
        uy = 233.0483557773859
        fx = 538.391033533567
        fy = 538.085452058436
        i_fx = 1 / fx
        i_fy = 1 / fy
        _, depth_image = get_images(os.path.join(self.rgb_directory, str(frame_id).zfill(4) + '.png'),
                                    os.path.join(self.depth_directory, str(frame_id).zfill(4) + '.png'),
                                    self.pix2pose)
        points_3D = np.zeros((depth_image.shape[0] * depth_image.shape[1], 3))
        p_count = 0
        for u in range(depth_image.shape[0]):
            for v in range(depth_image.shape[1]):
                points_3D[p_count, 2] = depth_image[u, v]
                points_3D[p_count, 1] = (u - ux) * points_3D[p_count, 2] * i_fx
                points_3D[p_count, 0] = (v - uy) * points_3D[p_count, 2] * i_fy
                p_count += 1
        #coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        #points_3D = points_3D.dot(coord_change_mat.T)

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(points_3D)
        scene_pcd.paint_uniform_color([0.5, 0.5, 0.9])
        vis.add_geometry(scene_pcd)

        # End
        vis.run()
        vis.destroy_window()

    def save_to_file(self, transform):
        save_path = copy.deepcopy(self.dataset_directory)
        if save_path[-1] == '/':
            save_path = save_path[0:-1]
        save_filename = save_path + '.npy'
        print('Saving transform\n{}\nto file {}'.format(transform, save_filename))
        np.save(save_filename, transform)

    def combine_grasps(self):
        # Get the name of the directory to search
        dir_path = copy.deepcopy(self.dataset_directory)
        if dir_path[-1] == '/':
            dir_path = save_path[0:-1]
        slash_idx = dir_path.rfind('/')
        dir_path = dir_path[0:slash_idx]

        # Get all grasp files
        filelist = []
        for f in os.listdir(dir_path):
            if f.endswith('.npy'):
                filelist.append(os.path.join(dir_path, f))
        filelist.sort()

        # Load each file and combine into one grasp file
        all_grasps = []
        for f in filelist:
            print(f)
            all_grasps.append(np.load(f))

        # Save as a new grasp file
        save_filename = dir_path + '.npy'
        print('Saving grasp transformations\n{}\nto file {}'.format(np.asarray(all_grasps), save_filename))
        np.save(save_filename, np.array(all_grasps))


class HandPoseEstimator:
    def __init__(self, args):
        self.proto_file = args.proto_file
        self.weights_file = args.weights_file
        self.pix2pose = args.pix2pose
        self.width = 640
        self.height = 480
        self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)

    def process(self, rgb_image, depth_image):
        # Get image properties
        self.width = rgb_image.shape[1]
        self.height = rgb_image.shape[0]
        aspect_ratio = self.width / self.height

        # Input image dimensions for the network
        in_height = 368
        in_width = int(((aspect_ratio * in_height) * 8) // 8)
        net_input = cv2.dnn.blobFromImage(rgb_image, 1.0 / 255, (in_width, in_height), (0, 0, 0),
                                          swapRB=False, crop=False)

        # Get the prediction
        self.net.setInput(net_input)
        pred = self.net.forward()

        # Get the 3D coordinates
        points = self.get_keypoints(pred)
        points3D = self.get_world_coordinates(points, depth_image)
        if not self.pix2pose:
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

    def get_world_coordinates(self, points, depth_image):
        points_3D = np.zeros((len(points), 3))

        if self.pix2pose:
            ux = 315.3074696331638
            uy = 233.0483557773859
            fx = 538.391033533567
            fy = 538.085452058436
        else:
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


def get_images(rgb_filename, dep_filename, ros_image=False):
    rgb_image = cv2.imread(rgb_filename)
    if ros_image:
        dep_image = cv2.imread(dep_filename, cv2.CV_16UC1).astype(np.int16)
        depth_scale = 0.001
    else:
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


def load_offset(offsets_file, dataset_path):
    x = dataset_path.split('/')
    object_name = x[-2]
    bag_name = x[-1]
    print('Object: {}  Bag: {}'.format(object_name, bag_name))

    offset = 0.045
    f = open(offsets_file, "r")
    for line in f:
        split_line = line.split()
        x = split_line[0].split('/')
        if x[-2] == object_name and x[-1] == bag_name:
            offset = float(split_line[1])
            print('Found offset of {}'.format(offset))
            break
    f.close()

    return offset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--pointnet_model', type=str, required=True, help='path to the pointnet model to load')
    #parser.add_argument('--hand', type=str, required=True, help="path to the hand pose estimates")
    #parser.add_argument('--object', type=str, required=True, help="path to the object pose estimates")
    #parser.add_argument('--object_model', type=str, required=True,
    #                    help="path to the directory that stores the object model")

    opt = parser.parse_args()
    opt.pointnet_model = '/home/tpatten/Data/ICAS2020/Models/ALL_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_149.pth'
    opt.object_model = ''
    opt.proto_file = '/home/tpatten/Code/hand-tracking/caffe_models/pose_deploy.prototxt'
    opt.weights_file = '/home/tpatten/Code/hand-tracking/caffe_models/pose_iter_102000.caffemodel'
    opt.verbose = True
    '''
    opt.target = 'ABF10'
    opt.hand_offset = 0.045
    opt.start_frame = 0
    opt.dataset = '/home/tpatten/Data/Hands/HO3D/train'
    opt.ground_truth_prefix = 'meta/grasp_bl_'
    opt.alternate_start = False
    opt.pix2pose = False
    opt.save = False
    '''
    #'''
    opt.target = ''
    opt.ground_truth_prefix = ''
    opt.start_frame = 0
    opt.dataset = '/home/tpatten/Data/ICAS2020/Rosbags/035_power_drill/2020-06-30-14-21-30'
    opt.hand_offset = load_offset('/home/tpatten/Data/ICAS2020/Rosbags/offsets.txt', opt.dataset)
    opt.alternate_start = False
    opt.pix2pose = True
    opt.save = False
    opt.combine_grasp_files = True
    #'''

    if opt.target == 'BB10':
        opt.start_frame = 502

    if opt.alternate_start:
        if opt.target == 'BB10':
            opt.start_frame = 973
        elif opt.target == 'GPMF10':
            opt.start_frame = 412
        elif opt.target == 'GSF10':
            opt.start_frame = 614
        elif opt.target == 'MDF10':
            opt.start_frame = 1005

    print(opt)

    learner = GraspLearner(opt)

    # Files
    # ---
    # 004_sugar_box/2020-06-16-11-56-35
    # 004_sugar_box/2020-06-16-14-38-15
    # 004_sugar_box/2020-06-16-15-01-01
    # 004_sugar_box/2020-06-16-15-01-39
    # ---
    # 005_tomato_soup_can/2020-06-16-11-46-35
    # 005_tomato_soup_can/2020-06-16-11-47-00
    # 005_tomato_soup_can/2020-06-16-14-41-58
    # 005_tomato_soup_can/2020-06-16-14-42-26
    # ---
    # 006_mustard_bottle/2020-06-16-11-48-42
    # 006_mustard_bottle/2020-06-16-14-24-59
    # 006_mustard_bottle/2020-06-16-14-25-29
    # 006_mustard_bottle/2020-06-16-14-44-13
    # ---
    # 025_mug/2020-06-16-11-52-00
    # 025_mug/2020-06-16-11-52-25
    # 025_mug/2020-06-16-14-28-01
    # 025_mug/2020-06-16-14-46-37
    # ---
    # 035_power_drill/2020-06-16-11-54-44
    # 035_power_drill/2020-06-16-14-36-06
    # 035_power_drill/2020-06-16-14-59-07
    # ---
