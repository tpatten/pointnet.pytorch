from __future__ import print_function
import argparse
import os
import random
import numpy as np
import error_def
import pickle

gripper_diameter = 0.232280153674483
num_steps = 15

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='results/results.pkl', help='pickle file with stored results')

opt = parser.parse_args()
print(opt)

add_th_base = np.linspace(0, 1.0, num_steps, endpoint=True, dtype=np.float32)
add_th = gripper_diameter * add_th_base
adds_th = gripper_diameter * add_th_base
tr_th_base = np.linspace(0, 0.5, num_steps, endpoint=True, dtype=np.float32)
t_th = tr_th_base
r_th = np.radians(100*tr_th_base)

with open(opt.file, 'rb') as f:
    try:
        pickle_data = pickle.load(f, encoding='latin1')
    except:
        pickle_data = pickle.load(f)
all_errors = pickle_data

if error_def.ADD_CODE in all_errors.keys():
    eval_add = []
    for t in add_th:
        e = error_def.eval_add_adds(all_errors[error_def.ADD_CODE], t)
        eval_add.append(e)
    print('--- ADD ---')
    for e in eval_add:
        print(e)
    print('')

if error_def.ADDS_CODE in all_errors.keys():
    eval_adds = []
    for t in adds_th:
        e = error_def.eval_add_adds(all_errors[error_def.ADDS_CODE], t)
        eval_adds.append(e)
    print('--- ADD SYMMETRIC ---')
    for e in eval_adds:
        print(e)
    print('')

if error_def.TRANSLATION_CODE in all_errors.keys():
    t_e = all_errors[error_def.TRANSLATION_CODE]
    r_e = []
    r_e_x = []
    r_e_y = []
    r_e_z = []
    if error_def.ROTATION_CODE in all_errors.keys():
        r_e = all_errors[error_def.ROTATION_CODE]
    if error_def.ROTATION_X_CODE in all_errors.keys() and \
            error_def.ROTATION_Y_CODE in all_errors.keys() and error_def.ROTATION_Z_CODE in all_errors.keys():
        r_e_x = all_errors[error_def.ROTATION_X_CODE]
        r_e_y = all_errors[error_def.ROTATION_Y_CODE]
        r_e_z = all_errors[error_def.ROTATION_Z_CODE]
    eval_tr = []
    eval_trxyz = []
    for i in range(len(t_th)):
        e = error_def.eval_translation_rotation(t_e, r_e, r_e_x, r_e_y, r_e_z, t_th[i], r_th[i])
        eval_tr.append(e[0])
        eval_trxyz.append(e[1])
    if len(r_e) > 0:
        print('--- T/R ---')
        for e in eval_tr:
            print(e)
        print('')
    if len(r_e_x) > 0:
        print('--- T/Rxyz ---')
        for e in eval_trxyz:
            print(e)
        print('')
