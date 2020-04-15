from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import pickle
import copy
import random
import transforms3d as tf3d


LEFT_TIP_IN_CLOUD = 3221
RIGHT_TIP_IN_CLOUD = 5204


def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 regression=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.regression = regression
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        centroid = np.mean(point_set, axis=0)
        # print(point_set.shape, seg.shape)
        # print(centroid.shape, centroid)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        centroid = torch.from_numpy(centroid)

        if self.regression:
            return point_set, centroid
        elif self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


class HO3DDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 gripper_xyz='hand_open.xyz',
                 data_augmentation=True,
                 subset_name='ALL',
                 randomly_flip_closing_angle=False,
                 center_to_wrist_joint=False,
                 disable_global_augmentation=False):
        self.root = root
        splitfile, split_root_name = self.get_split_file(split, subset_name)
        self.data_augmentation = data_augmentation
        self.randomly_flip_closing_angle = randomly_flip_closing_angle
        self.center_to_wrist_joint = center_to_wrist_joint
        self.disable_global_augmentation = disable_global_augmentation

        # Get the gripper model
        self.base_pcd = np.loadtxt(gripper_xyz)

        # Get the files
        f = open(splitfile, "r")
        filelist = [line[:-1] for line in f]
        f.close()

        # Shuffle the files
        random.shuffle(filelist)

        # Create the data path object
        self.datapath = []
        for file in filelist:
            subject, seq = file.split('/')
            joints_filename = os.path.join(self.root, split_root_name, subject, 'meta', seq + '.pkl')
            grasp_pose_filename = os.path.join(self.root, split_root_name, subject, 'meta', 'grasp_bl_' + seq + '.pkl')
            self.datapath.append((joints_filename, grasp_pose_filename))

    def __getitem__(self, index):
        fn = self.datapath[index]

        # Get the hand joints
        with open(fn[0], 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)
        point_set = pickle_data['handJoints3D']

        # Get the gripper pose
        with open(fn[1], 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)
        grasp_pose = pickle_data.reshape(4, 4)
        # Rotate the cloud
        pts = copy.deepcopy(self.base_pcd)
        pts = np.matmul(pts, np.linalg.inv(grasp_pose[:3, :3]))

        # Get the approach and roll
        left_tip = pts[LEFT_TIP_IN_CLOUD]
        right_tip = pts[RIGHT_TIP_IN_CLOUD]
        gripper_mid_point = 0.5 * (left_tip + right_tip)
        approach_vec = np.asarray(gripper_mid_point)
        approach_vec /= np.linalg.norm(approach_vec)
        close_vec = right_tip - left_tip
        close_vec /= np.linalg.norm(close_vec)
        if self.randomly_flip_closing_angle and np.random.random_sample() > 0.5:
            close_vec *= -1

        # Create the target
        target = np.zeros((1, 9)).flatten()
        target[0:3] = grasp_pose[:3, 3].flatten()
        target[3:6] = approach_vec
        target[6:9] = close_vec

        # center the joints
        if self.center_to_wrist_joint:
            offset = np.copy(point_set[0, :]).reshape((1, 3))
        else:
            offset = np.expand_dims(np.mean(point_set, axis=0), 0)
        point_set = point_set - offset
        # scale the joints
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        # center and scale the grasp pose
        target[0:3] -= offset.flatten()
        target[0:3] /= dist

        # Apply augmentation
        if self.data_augmentation:
            # theta = np.random.uniform(-0.1 * np.pi, 0.1 * np.pi)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            # point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
            point_set, target = self.augment_data(point_set, target, self.disable_global_augmentation)

        # Create tensors and return
        point_set = torch.from_numpy(point_set.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))

        return point_set, target, offset, dist

    def __len__(self):
        return len(self.datapath)

    def get_split_file(self, split, subset_name):
        subset_name_up = subset_name.upper()
        if subset_name_up == 'SHSU':
            subset_name_up = 'ShSu'
        elif subset_name_up == 'XSHSU':
            subset_name_up = 'xShSu'

        suffix = 'grasp_train.txt'
        if split == 'test':
            suffix = 'grasp_test.txt'

        if subset_name_up[0] == 'X' and split == 'test':
            subset_name_up = subset_name_up[1:]

        if subset_name_up == 'ALL':
            splitfile = os.path.join(self.root, 'splits_new', suffix)
        else:
            splitfile = os.path.join(self.root, 'splits_new', subset_name_up + '_' + suffix)

        split_root_name = 'train'

        return splitfile, split_root_name

    def augment_data(self, point_set, target, disable_global=False):
        # Global rotation
        rotation_matrix = tf3d.euler.euler2mat(np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi, np.pi))
        point_set_augmented = np.copy(point_set)
        if not disable_global:
            point_set_augmented = np.matmul(point_set_augmented, rotation_matrix)

        target_augmented = np.copy(target)
        target_augmented = target_augmented.reshape((3, 3))
        if not disable_global:
            target_augmented = np.matmul(target_augmented, rotation_matrix)

        # Local rotation and jitter to point set
        theta_lims = [-0.025 * np.pi, 0.025 * np.pi]
        jitter_scale = 0.01
        rotation_matrix = tf3d.euler.euler2mat(np.random.uniform(theta_lims[0], theta_lims[1]),
                                               np.random.uniform(theta_lims[0], theta_lims[1]),
                                               np.random.uniform(theta_lims[0], theta_lims[1]))
        point_set_augmented = np.matmul(point_set_augmented, rotation_matrix)  # random rotation
        jitter = np.random.normal(0, jitter_scale, size=point_set_augmented.shape)  # random jitter
        point_set_augmented += jitter

        if self.center_to_wrist_joint:
            offset_augmented = np.copy(point_set_augmented[0, :]).reshape((1, 3))
            point_set_augmented -= offset_augmented
            target_augmented[0, :] -= offset_augmented.flatten()

        target_augmented = target_augmented.reshape((1, 9)).flatten()

        return point_set_augmented, target_augmented


if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root=datapath, class_choice=['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(), seg.type())

        d = ShapeNetDataset(root=datapath, classification=True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(), cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

