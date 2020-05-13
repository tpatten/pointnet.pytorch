from __future__ import print_function
import argparse
import os
import sys
import random
from pointnet.model import parse_model_filename, load_regression_model
import torch
import numpy as np
import pickle
import copy
import error_def
import matplotlib.pyplot as plt


PASSGREEN = lambda x: '\033[92m' + x + '\033[0m'
FAILRED = lambda x: '\033[91m' + x + '\033[0m'
DIAMETER = 0.232280153674483
CAMERAS = ['10', '11', '12', '13', '14']


def evaluate(reg_model, dataset_path, data_subset, use_loaded_data=False):
    # Load the hand tracker data
    hand_tracker_error_filename = os.path.join(dataset_path, 'results/hand_tracker_errors.pkl')
    with open(hand_tracker_error_filename, 'rb') as f:
        try:
            hand_errors = pickle.load(f, encoding='latin1')
        except:
            hand_errors = pickle.load(f)

    # Load the existing grasp pose errors
    save_filename = os.path.join(dataset_path, 'results/grasp_pose_errors.pkl')
    if not os.path.exists(save_filename):
        with open(save_filename, 'wb') as f:
            pickle.dump(hand_errors, f)

    with open(save_filename, 'rb') as f:
        try:
            grasp_errors = pickle.load(f, encoding='latin1')
        except:
            grasp_errors = pickle.load(f)

    # Get the subject name
    subset_name = data_subset
    if subset_name[0] == 'X':
        subset_name = subset_name[1:]

    # Compute the grasp pose errors
    if not use_loaded_data:
        # Create subject and camera dictionary and get the number of frames
        sub_cam_dict = {}
        num_frames = 0
        for i in range(len(hand_errors)):
            if hand_errors[i][0] == subset_name:
                if hand_errors[i][2].shape[0] > 0:
                    num_frames = hand_errors[i][2].shape[0]
                    sub_cam_dict[subset_name + hand_errors[i][1]] = i
                else:
                    sub_cam_dict[subset_name + hand_errors[i][1]] = None

        # For each file
        gripper_pts = np.loadtxt('hand_open_symmetric.xyz')
        for c in CAMERAS:
            subject = subset_name + c
            for i in range(num_frames):
                seq = str(i).zfill(4)
                hand_filename = os.path.join(dataset_path, 'train', subject, 'hand_tracker', seq + '.pkl')
                if os.path.exists(hand_filename):
                    grasp_pose_filename = os.path.join(dataset_path, 'train', subject, 'meta',
                                                       'grasp_bl_' + seq + '.pkl')
                    print('\n--- Processing file {} ---'.format(hand_filename))
                    # Evaluate
                    err = compute_error(reg_model, hand_filename, grasp_pose_filename, gripper_pts)
                else:
                    print('No hand estimate for {}'.format(hand_filename))
                    err = np.nan
                idx = sub_cam_dict[subject]
                if idx is not None:
                    grasp_errors[idx][2][i] = err

        # Save the processed files
        if not opt.use_loaded_data:
            with open(save_filename, 'wb') as f:
                pickle.dump(grasp_errors, f)

    # Plot the errors
    plot_errors(hand_errors, grasp_errors, target=subset_name)


def compute_error(reg_model, hand_filename, grasp_pose_filename, gripper_pts):
    # Load the points
    with open(hand_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    point_set = pickle_data['handJoints3D'].astype(np.float32)
    point_set = np.nan_to_num(point_set)

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
    pred = reg_model(points_batch)
    pred_np = pred.data.cpu().numpy()[0]

    # Load the ground truth grasp pose
    with open(grasp_pose_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    gripper_transform = pickle_data.reshape(4, 4)

    # Transform with the prediction
    predicted_gripper_transform = error_def.to_transform(pred_np, offset, dist)

    print('ADD   {:.4f}'.format(error_def.add_error(gripper_transform, predicted_gripper_transform, gripper_pts)))
    adds_error = error_def.add_symmetric_error(gripper_transform, predicted_gripper_transform, gripper_pts)
    print('ADDS  {:.4f}'.format(adds_error))
    print('Tra   {:.2f}'.format(error_def.translation_error(gripper_transform, predicted_gripper_transform) * 100))
    print('Rot   {:.2f}'.format(np.degrees(error_def.rotation_error(gripper_transform,
                                                                    predicted_gripper_transform)[0])))
    if adds_error < 0.1 * DIAMETER:
        print('10%%:  %s' % (PASSGREEN('PASS')))
    else:
        print('10%%:  %s' % (FAILRED('FAIL')))
    if adds_error < 0.2 * DIAMETER:
        print('20%%:  %s' % (PASSGREEN('PASS')))
    else:
        print('20%%:  %s' % (FAILRED('FAIL')))

    return adds_error


def plot_errors(h_errors, g_errors, target=''):
    x_vals = []
    y_vals = []

    if target == '':
        for i in range(len(h_errors)):
            x_vals.extend(h_errors[i][2])
            y_vals.extend(g_errors[i][2])

    for i in range(len(h_errors)):
        if h_errors[i][0] == target:
            x_vals.extend(h_errors[i][2])
            y_vals.extend(g_errors[i][2])

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_vals, list(np.asarray(y_vals) * 100))
    ax.set_title(target)
    ax.set_xlabel('Hand mean joint position error (cm)')
    ax.set_ylabel('Grasp ADD-S error (cm)')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")

    opt = parser.parse_args()
    opt.use_loaded_data = True
    f_args = parse_model_filename(os.path.basename(opt.model))
    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Load the network
    regressor = load_regression_model(f_args['arch'], 9, f_args['dropout_p'], f_args['average_pool'], model=opt.model)
    regressor = regressor.eval()

    # Evaluate
    evaluate(regressor, opt.dataset, f_args['data_subset'], use_loaded_data=opt.use_loaded_data)
