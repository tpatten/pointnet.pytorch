from __future__ import print_function
import argparse
import os
import random
import numpy as np
import open3d as o3d
import pickle
import transforms3d as tf3d
import copy
import cv2
import error_def

LEFT_TIP_IN_CLOUD = 3221
RIGHT_TIP_IN_CLOUD = 5204
global_augmentation = False
local_augmentation = False
theta_lims = [-0.05 * np.pi, 0.05 * np.pi]
jitter_scale = 0.01
center_to_wrist_joint = True


def visualize(meta_filename, grasp_pose_filename, models_path):
    # Load the points and object model
    with open(meta_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    point_set = pickle_data['handJoints3D'].astype(np.float32)
    # center
    if center_to_wrist_joint:
        offset = np.copy(point_set[0, :]).reshape((1, 3))
    else:
        offset = np.expand_dims(np.mean(point_set, axis=0), 0)
    point_set -= offset
    # scale
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set /= dist

    dist_original = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    print('Original: {} {}'.format(offset, dist_original))

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
    pts -= offset.flatten()
    pts /= dist
    obj_pcd.points = o3d.utility.Vector3dVector(pts)

    # Load the ground truth grasp pose
    with open(grasp_pose_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    gripper_transform = pickle_data.reshape(4, 4)

    # Load the gripper cloud
    gripper_pcd = o3d.geometry.PointCloud()
    gripper_pcd.points = o3d.utility.Vector3dVector(np.loadtxt('hand_open.xyz'))

    # Rotate
    pts = np.asarray(gripper_pcd.points)
    pts = np.matmul(pts, np.linalg.inv(gripper_transform[:3, :3]))

    # Compute the target vectors
    left_tip = pts[LEFT_TIP_IN_CLOUD]
    right_tip = pts[RIGHT_TIP_IN_CLOUD]
    gripper_mid_point = 0.5 * (left_tip + right_tip)
    approach_vec = np.asarray(gripper_mid_point)
    approach_vec /= np.linalg.norm(approach_vec)
    close_vec = right_tip - left_tip
    close_vec /= np.linalg.norm(close_vec)
    gripper_centroid = gripper_transform[:3, 3].flatten()
    gripper_centroid -= offset.flatten()
    gripper_centroid /= dist
    print('Target values')
    print('Centroid: {}'.format(gripper_centroid))
    print('Approach: {}'.format(approach_vec))
    print('Close: {}'.format(close_vec))

    # Now translate the point cloud and modify it
    pts += gripper_transform[:3, 3]
    pts -= offset.flatten()
    pts /= dist
    gripper_pcd.points = o3d.utility.Vector3dVector(pts)

    # Augmentation
    '''
    point_set_augmented = np.copy(point_set)
    gripper_pcd_augmented = copy.deepcopy(gripper_pcd)
    gripper_centroid_augmented = np.copy(gripper_centroid).reshape((1, 3))
    approach_vec_augmented = np.copy(approach_vec).reshape((1, 3))
    close_vec_augmented = np.copy(close_vec).reshape((1, 3))
    if global_augmentation:
        #theta = np.random.uniform(-np.pi, np.pi)
        #print('Global theta augmentation: {}'.format(np.degrees(theta)))
        #rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        #point_set_augmented[:, [0, 2]] = point_set_augmented[:, [0, 2]].dot(rotation_matrix)
        #pts = np.asarray(gripper_pcd_augmented.points)
        #pts[:, [0, 2]] = pts[:, [0, 2]].dot(rotation_matrix)
        #gripper_pcd_augmented.points = o3d.utility.Vector3dVector(pts)
        #gripper_centroid_augmented[:, [0, 2]] = gripper_centroid_augmented[:, [0, 2]].dot(rotation_matrix)
        #approach_vec_augmented[:, [0, 2]] = approach_vec_augmented[:, [0, 2]].dot(rotation_matrix)
        #close_vec_augmented[:, [0, 2]] = close_vec_augmented[:, [0, 2]].dot(rotation_matrix)
        rotation_matrix = tf3d.euler.euler2mat(np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi, np.pi))
        point_set_augmented = point_set_augmented.dot(rotation_matrix)
        pts = np.asarray(gripper_pcd_augmented.points)
        pts = pts.dot(rotation_matrix)
        gripper_pcd_augmented.points = o3d.utility.Vector3dVector(pts)
        gripper_centroid_augmented = gripper_centroid_augmented.dot(rotation_matrix)
        approach_vec_augmented = approach_vec_augmented.dot(rotation_matrix)
        close_vec_augmented = close_vec_augmented.dot(rotation_matrix)
    gripper_centroid_augmented = gripper_centroid_augmented.T.flatten()
    approach_vec_augmented = approach_vec_augmented.T.flatten()
    close_vec_augmented = close_vec_augmented.T.flatten()

    if local_augmentation:
        #theta = np.random.uniform(theta_lims[0], theta_lims[1])
        #rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        #point_set_augmented[:, [0, 2]] = point_set_augmented[:, [0, 2]].dot(rotation_matrix)  # random rotation
        rotation_matrix = tf3d.euler.euler2mat(np.random.uniform(theta_lims[0], theta_lims[1]),
                                               np.random.uniform(theta_lims[0], theta_lims[1]),
                                               np.random.uniform(theta_lims[0], theta_lims[1]))
        point_set_augmented = point_set_augmented.dot(rotation_matrix)  # random rotation
        jitter = np.random.normal(0, jitter_scale, size=point_set_augmented.shape)  # random jitter
        point_set_augmented += jitter
    '''

    point_set_augmented = np.copy(point_set)
    gripper_pcd_augmented = copy.deepcopy(gripper_pcd)
    target = np.zeros((1, 9)).flatten()
    target[0:3] = np.copy(gripper_centroid)
    target[3:6] = np.copy(approach_vec)
    target[6:9] = np.copy(close_vec)
    target = target.reshape((3, 3))
    if global_augmentation:
        rotation_matrix = tf3d.euler.euler2mat(np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi, np.pi))
        point_set_augmented = np.matmul(point_set_augmented, rotation_matrix)
        pts = np.asarray(gripper_pcd_augmented.points)
        pts = pts.dot(rotation_matrix)
        gripper_pcd_augmented.points = o3d.utility.Vector3dVector(pts)
        target = np.matmul(target, rotation_matrix)
    gripper_centroid_augmented = target[0, :].T.flatten()
    approach_vec_augmented = target[1, :].T.flatten()
    close_vec_augmented = target[2, :].T.flatten()

    if local_augmentation:
        rotation_matrix = tf3d.euler.euler2mat(np.random.uniform(theta_lims[0], theta_lims[1]),
                                               np.random.uniform(theta_lims[0], theta_lims[1]),
                                               np.random.uniform(theta_lims[0], theta_lims[1]))
        point_set_augmented = np.matmul(point_set_augmented, rotation_matrix)  # random rotation
        jitter = np.random.normal(0, jitter_scale, size=point_set_augmented.shape)  # random jitter
        point_set_augmented += jitter

    if center_to_wrist_joint:
        offset_augmented = np.copy(point_set[0, :]).reshape((1, 3))
        point_set_augmented -= offset_augmented
        gripper_centroid_augmented -= offset_augmented.flatten()

    offset_augmented = np.expand_dims(np.mean(point_set_augmented, axis=0), 0)
    dist_augmented = np.max(np.sqrt(np.sum(point_set_augmented ** 2, axis=1)), 0)
    print('Augmented: {} {}'.format(offset_augmented, dist_augmented))

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Object
    obj_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(obj_pcd)

    # Original
    gripper_pcd.paint_uniform_color([0.2, 0.8, 0.2])
    vis.add_geometry(gripper_pcd)

    for i in range(point_set.shape[0]):
        mm = o3d.create_mesh_sphere(radius=0.05)
        mm.compute_vertex_normals()
        mm.paint_uniform_color([0., 1., 0.])
        pos = point_set[i, :]
        tt = np.eye(4)
        tt[0:3, 3] = pos
        mm.transform(tt)
        vis.add_geometry(mm)

    line_points = [gripper_centroid, gripper_centroid + approach_vec, gripper_centroid + close_vec]
    lines = [[0, 1], [0, 2]]
    line_colors = [[0.2, 0.8, 0.2], [0.2, 0.8, 0.2]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    vis.add_geometry(line_set)

    mm = o3d.create_mesh_sphere(radius=0.025)
    mm.compute_vertex_normals()
    mm.paint_uniform_color([0.8, 0.8, 0.2])
    tt = np.eye(4)
    tt[0:3, 3] = gripper_centroid
    mm.transform(tt)
    vis.add_geometry(mm)

    pts = np.asarray(gripper_pcd.points)
    left_tip = pts[LEFT_TIP_IN_CLOUD]
    right_tip = pts[RIGHT_TIP_IN_CLOUD]
    gripper_mid_point = 0.5 * (left_tip + right_tip)
    mm = o3d.create_mesh_sphere(radius=0.025)
    mm.compute_vertex_normals()
    mm.paint_uniform_color([0.8, 0.8, 0.2])
    tt = np.eye(4)
    tt[0:3, 3] = gripper_mid_point
    mm.transform(tt)
    vis.add_geometry(mm)

    # Augmented
    # aug_offset = np.asarray([1, 0, 0])
    aug_offset = np.asarray([0, 0, 0])
    gripper_pcd_augmented.paint_uniform_color([0.2, 0.2, 0.8])
    pts = np.asarray(gripper_pcd_augmented.points)
    pts += aug_offset
    gripper_pcd_augmented.points = o3d.utility.Vector3dVector(pts)
    vis.add_geometry(gripper_pcd_augmented)

    for i in range(point_set_augmented.shape[0]):
        mm = o3d.create_mesh_sphere(radius=0.05)
        mm.compute_vertex_normals()
        mm.paint_uniform_color([0., 0., 1.])
        pos = point_set_augmented[i, :] + aug_offset
        tt = np.eye(4)
        tt[0:3, 3] = pos
        mm.transform(tt)
        vis.add_geometry(mm)

    gripper_centroid_augmented += aug_offset
    line_points = [gripper_centroid_augmented, gripper_centroid_augmented + approach_vec_augmented,
                   gripper_centroid_augmented + close_vec_augmented]
    lines = [[0, 1], [0, 2]]
    line_colors = [[0.2, 0.2, 0.8], [0.2, 0.2, 0.8]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    vis.add_geometry(line_set)

    mm = o3d.create_mesh_sphere(radius=0.025)
    mm.compute_vertex_normals()
    mm.paint_uniform_color([0.8, 0.2, 0.8])
    tt = np.eye(4)
    tt[0:3, 3] = gripper_centroid_augmented
    mm.transform(tt)
    vis.add_geometry(mm)

    pts = np.asarray(gripper_pcd_augmented.points)
    left_tip = pts[LEFT_TIP_IN_CLOUD]
    right_tip = pts[RIGHT_TIP_IN_CLOUD]
    gripper_mid_point_augmented = 0.5 * (left_tip + right_tip)
    mm = o3d.create_mesh_sphere(radius=0.025)
    mm.compute_vertex_normals()
    mm.paint_uniform_color([0.8, 0.2, 0.8])
    tt = np.eye(4)
    tt[0:3, 3] = gripper_mid_point_augmented
    mm.transform(tt)
    vis.add_geometry(mm)

    # Coordinate frame
    vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.8))

    # End
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="meta file to observe")

    opt = parser.parse_args()
    opt.models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'

    random.seed()

    # Get the files
    meta_filename = opt.file
    idx = meta_filename.rfind('/')
    grasp_pose_filename = meta_filename[0:idx + 1] + 'grasp_bl_' + meta_filename[idx + 1:]

    print(' --- Processing file ---\n{}\n{}'.format(meta_filename, grasp_pose_filename))

    # Visualize
    visualize(meta_filename, grasp_pose_filename, opt.models_path)
