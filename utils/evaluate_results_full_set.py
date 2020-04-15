from __future__ import print_function
import argparse
import os
from os.path import isfile, join, splitext
import random
import numpy as np
import error_def
import pickle

gripper_diameter = 0.232280153674483
num_steps = 15
add_lim = 0.5
save_all_errors = False

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='results', help='directory where all the set of results are stored')

opt = parser.parse_args()
print(opt)

add_th_base = np.linspace(0, add_lim, num_steps, endpoint=True, dtype=np.float32)
add_th = gripper_diameter * add_th_base
adds_th = gripper_diameter * add_th_base
tr_th_base = np.linspace(0, 0.5, num_steps, endpoint=True, dtype=np.float32)
t_th = tr_th_base
r_th = np.radians(100*tr_th_base)

all_errors = {}
all_errors[error_def.ADD_CODE] = []
all_errors[error_def.ADDS_CODE] = []
all_errors[error_def.TRANSLATION_CODE] = []
all_errors[error_def.ROTATION_CODE] = []
all_errors[error_def.ROTATION_X_CODE] = []
all_errors[error_def.ROTATION_Y_CODE] = []
all_errors[error_def.ROTATION_Z_CODE] = []

# Accumulate all the errors from all the sets
pkl_files = [f for f in os.listdir(opt.dir) if isfile(join(opt.dir, f)) and splitext(f)[1] == '.pkl']
for pklf in pkl_files:
    print(pklf)
    with open(join(opt.dir, pklf), 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    errs = pickle_data

    for key in all_errors:
        all_errors[key].extend(errs[key])

print('--- ADD/S THRESHOLDS ---')
for t in add_th:
    print(t)
print('')

print('--- T/R THRESHOLDS ---')
for t in t_th:
    print(t)
print('')

eval_add = []
for t in add_th:
    e = error_def.eval_add_adds(all_errors[error_def.ADD_CODE], t)
    eval_add.append(e[0])
print('--- ADD ---')
for e in eval_add:
    print(e)
print('')

eval_adds = []
for t in adds_th:
    e = error_def.eval_add_adds(all_errors[error_def.ADDS_CODE], t)
    eval_adds.append(e[0])
print('--- ADD SYMMETRIC ---')
for e in eval_adds:
    print(e)
print('')

t_e = all_errors[error_def.TRANSLATION_CODE]
r_e = all_errors[error_def.ROTATION_CODE]
r_e_x = all_errors[error_def.ROTATION_X_CODE]
r_e_y = all_errors[error_def.ROTATION_Y_CODE]
r_e_z = all_errors[error_def.ROTATION_Z_CODE]
eval_tr = []
eval_trxyz = []
for i in range(len(t_th)):
    e = error_def.eval_translation_rotation(t_e, r_e, r_e_x, r_e_y, r_e_z, t_th[i], r_th[i])
    eval_tr.append(e[0][0])
    eval_trxyz.append(e[1][0])
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

print('--- AVERAGE ERRORS ---')
print('ADD\tADDS\tT\tRx\tRy\tRz\tR')
print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
    np.mean(np.asarray(all_errors[error_def.ADD_CODE])),
    np.mean(np.asarray(all_errors[error_def.ADDS_CODE])),
    np.mean(np.asarray(all_errors[error_def.TRANSLATION_CODE])) * 1000,
    np.degrees(np.mean(np.asarray(all_errors[error_def.ROTATION_X_CODE]))),
    np.degrees(np.mean(np.asarray(all_errors[error_def.ROTATION_Y_CODE]))),
    np.degrees(np.mean(np.asarray(all_errors[error_def.ROTATION_Z_CODE]))),
    np.degrees(np.mean(np.asarray(all_errors[error_def.ROTATION_CODE])))))
print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
    np.std(np.asarray(all_errors[error_def.ADD_CODE])),
    np.std(np.asarray(all_errors[error_def.ADDS_CODE])),
    np.std(np.asarray(all_errors[error_def.TRANSLATION_CODE])) * 1000,
    np.degrees(np.std(np.asarray(all_errors[error_def.ROTATION_X_CODE]))),
    np.degrees(np.std(np.asarray(all_errors[error_def.ROTATION_Y_CODE]))),
    np.degrees(np.std(np.asarray(all_errors[error_def.ROTATION_Z_CODE]))),
    np.degrees(np.std(np.asarray(all_errors[error_def.ROTATION_CODE])))))

if save_all_errors:
    with open(join(opt.dir, 'errors.pkl'), 'wb') as f:
        pickle.dump(all_errors, f)
