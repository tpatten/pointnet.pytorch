import numpy as np
import transforms3d as tf3d
import copy
import math


def to_transform(vec9, offset, dist):
    translation = copy.deepcopy(vec9[0:3])
    translation *= dist
    translation += offset.flatten()
    z_axis = copy.deepcopy(vec9[3:6])
    z_axis /= np.linalg.norm(z_axis)
    y_axis = copy.deepcopy(vec9[6:9])
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rot = np.eye(3)
    rot[:, 0] = x_axis
    rot[:, 1] = y_axis
    rot[:, 2] = z_axis

    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = translation

    return transform


def to_transform_batch(vecvec9, offsets, dists):
    transforms = []
    for i in range(len(vecvec9)):
        transforms.append(to_transform(vecvec9[i], offsets[i], dists[i]))
    return transforms


def translation_error(gt_pose, est_pose):
    return np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])


def rotation_error(gt_pose, est_pose):
    est_pose_new = copy.deepcopy(est_pose)
    rot_diff = np.abs(np.degrees(
        np.array(tf3d.euler.mat2euler(np.matmul(np.linalg.inv(gt_pose[:3, :3]), est_pose_new[:3, :3]), 'szyx'))))
    if rot_diff[0] > 90:
        est_pose_new[:3, 0] *= -1
        est_pose_new[:3, 1] *= -1
        rot_diff = np.abs(
            np.array(tf3d.euler.mat2euler(np.matmul(np.linalg.inv(gt_pose[:3, :3]), est_pose_new[:3, :3]), 'szyx')))
    else:
        rot_diff = np.radians(rot_diff)

    # Compute cosine or error
    error_cos = float(0.5 * (np.trace(est_pose_new[:3, :3].dot(np.linalg.inv(gt_pose[:3, :3]))) - 1.0))

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)

    return error, rot_diff[2], rot_diff[1], rot_diff[0]


def add_error(gt_pose, est_pose, pts):
    gt_pts = copy.deepcopy(pts)
    gt_pts = np.matmul(gt_pts, np.linalg.inv(gt_pose[:3, :3]))
    gt_pts += gt_pose[:3, 3]

    est_pts = copy.deepcopy(pts)
    est_pts = np.matmul(est_pts, np.linalg.inv(est_pose[:3, :3]))
    est_pts += est_pose[:3, 3]

    error = np.linalg.norm(est_pts - gt_pts, axis=1).mean()

    return error


def add_symmetric_error(gt_pose, est_pose, pts):
    error_1 = add_error(gt_pose, est_pose, pts)

    est_pose_rot_z = copy.deepcopy(est_pose)
    est_pose_rot_z[:3, 0] *= -1
    est_pose_rot_z[:3, 1] *= -1
    error_2 = add_error(gt_pose, est_pose_rot_z, pts)

    return min(error_1, error_2)


def get_all_errors(gt_pose, est_pose, pts):
    add = add_error(gt_pose, est_pose, pts)
    add_s = add_symmetric_error(gt_pose, est_pose, pts)
    t = translation_error(gt_pose, est_pose)
    r = rotation_error(gt_pose, est_pose)

    return add, add_s, t, r[0], r[1], r[2], r[3]


def get_all_errors_batch(gt_poses, est_poses, pts):
    errors = np.zeros((len(gt_poses), 7))
    for i in range(len(gt_poses)):
        errors[i, :] = get_all_errors(gt_poses[i], est_poses[i], pts)
    return errors


def eval_grasp(gt_pose, est_pose, th_trans=0.05, th_rot=15):  # 5cm, 15degree
    tra_diff = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    if tra_diff > th_trans:
        return False
    else:
        # rot_diff = np.matmul(np.linalg.inv(gt_pose), est_pose)
        rot_diff = np.abs(np.degrees(
            np.array(tf3d.euler.mat2euler(np.matmul(np.linalg.inv(gt_pose[:3, :3]), est_pose[:3, :3]), 'szyx'))))
        # -180~180
        print('Z', rot_diff[0])
        if rot_diff[1] < th_rot and rot_diff[2] < th_rot:  # ignore z_rotation of gripper (because of the symmetry)
            return True
        else:
            return False
