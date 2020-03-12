import numpy as np
import transforms3d as tf3d
import copy
import math


ADD_CODE = 'ADD'
ADDS_CODE = 'ADDS'
TRANSLATION_CODE = 'TRANSLATION'
ROTATION_CODE = 'ROTATION'
ROTATION_X_CODE = 'ROTATION_X'
ROTATION_Y_CODE = 'ROTATION_Y'
ROTATION_Z_CODE = 'ROTATION_Z'


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


def get_all_errors(gt_pose, est_pose, pts, error_dict):
    add = add_error(gt_pose, est_pose, pts)
    add_s = add_symmetric_error(gt_pose, est_pose, pts)
    t = translation_error(gt_pose, est_pose)
    r = rotation_error(gt_pose, est_pose)

    result_dict = {}
    for key in error_dict:
        if key == ADD_CODE:
            result_dict[ADD_CODE] = add
        elif key == ADDS_CODE:
            result_dict[ADDS_CODE] = add_s
        elif key == TRANSLATION_CODE:
            result_dict[TRANSLATION_CODE] = t
        elif key == ROTATION_CODE:
            result_dict[ROTATION_CODE] = r[0]
        elif key == ROTATION_X_CODE:
            result_dict[ROTATION_X_CODE] = r[1]
        elif key == ROTATION_Y_CODE:
            result_dict[ROTATION_Y_CODE] = r[2]
        elif key == ROTATION_Z_CODE:
            result_dict[ROTATION_Z_CODE] = r[3]

    return result_dict


def get_all_errors_batch(gt_poses, est_poses, pts, error_dict):
    results_dict = {}
    for key in error_dict:
        results_dict[key] = []
    for i in range(len(gt_poses)):
        errors = get_all_errors(gt_poses[i], est_poses[i], pts, error_dict)
        for key in error_dict:
            results_dict[key].append(errors[key])
    return results_dict


def eval_grasps(error_dict, add_th=None, adds_th=None, tr_th=None):
    add_eval = 0.
    adds_eval = 0.
    tr_eval = 0.
    trxyz_eval = 0.

    if add_th is not None and ADD_CODE in error_dict.keys():
        add_eval = eval_add_adds(error_dict[ADD_CODE], add_th)

    if adds_th is not None and ADDS_CODE in error_dict.keys():
        adds_eval = eval_add_adds(error_dict[ADDS_CODE], adds_th)

    if tr_th is not None:
        if TRANSLATION_CODE in error_dict.keys():
            t_e = error_dict[TRANSLATION_CODE]
            r_e = []
            r_e_x = []
            r_e_y = []
            r_e_z = []
            if ROTATION_CODE in error_dict.keys():
                r_e = error_dict[ROTATION_CODE]
            if ROTATION_X_CODE in error_dict.keys() and \
                    ROTATION_Y_CODE in error_dict.keys() and ROTATION_Z_CODE in error_dict.keys():
                r_e_x = error_dict[ROTATION_X_CODE]
                r_e_y = error_dict[ROTATION_Y_CODE]
                r_e_z = error_dict[ROTATION_Z_CODE]
            tr_eval, trxyz_eval = eval_translation_rotation(t_e, r_e, r_e_x, r_e_y, r_e_z, tr_th[0], tr_th[1])

    return add_eval, adds_eval, tr_eval, trxyz_eval


def eval_add_adds(errors, threshold):
    num_correct = 0
    for e in errors:
        if e <= threshold:
            num_correct += 1
    return float(num_correct) / float(len(errors))


def eval_translation_rotation(t_errors, r_errors, r_x_errors, r_y_errors, r_z_errors, t_threshold, r_threshold):
    if len(r_errors) > 0:
        num_correct = 0
        for i in range(len(t_errors)):
            if t_errors[i] <= t_threshold and r_errors[i] <= r_threshold:
                num_correct += 1
        tr_eval = float(num_correct) / float(len(t_errors))
    else:
        tr_eval = 0.

    if len(r_x_errors) > 0 and len(r_y_errors) > 0 and len(r_z_errors) > 0:
        num_correct = 0
        for i in range(len(t_errors)):
            if t_errors[i] <= t_threshold and r_x_errors[i] <= r_threshold and r_y_errors[i] <= r_threshold and \
                    r_z_errors[i] <= r_threshold:
                num_correct += 1
        trxyz_eval = float(num_correct) / float(len(t_errors))
    else:
        trxyz_eval = 0.

    return tr_eval, trxyz_eval
