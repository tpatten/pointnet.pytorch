from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils.error_def as error_def
from enum import IntEnum
import copy


MSE_LOSS_CODE = 'MSE'
L1_LOSS_CODE = 'L1'
SMOOTH_L1_LOSS_CODE = 'SMOOTHL1'
MODEL_LOSS_CODE = 'MODEL'


class Archs(IntEnum):
    PN = 1
    PN_Sym = 2
    PN_FC4 = 3
    PN_FC4_Sym = 4
    PN_FC45 = 5
    PN_FC45_Sym = 6
    PN_Small_3L = 7
    PN_Small_4L = 8
    PN_Half = 9
    PN_Half_FC4 = 10
    PN_FC4_256 = 11
    PN_LReLu = 12
    PN_Half_LReLu = 13
    PN_Flat = 14
    PN_NoPool = 15
    PN_NoPoolSmall = 16
    PN_Flat5Layer = 17
    PN_Flat_Split = 18


def parse_model_filename(filename):
    # Output is a dictionary
    f_args = {}
    sub_str = copy.deepcopy(filename)
    # Subset is the first
    idx = sub_str.find('_')
    f_args['data_subset'] = sub_str[0:idx]
    # Batch is the second
    idx = sub_str.find('batch')
    idx2 = sub_str.find('_', idx)
    f_args['batch_size'] = int(sub_str[idx + 5:idx2])
    # Epoch is the third
    idx = sub_str.find('nEpoch')
    idx2 = sub_str.find('_', idx)
    f_args['nEpoch'] = int(sub_str[idx + 6: idx2])

    # Get either modelLoss of regLoss
    idx = sub_str.find('regLoss')
    if idx != -1:
        f_args['model_loss'] = False
    else:
        f_args['model_loss'] = True

    # Check if dropout
    f_args['dropout_p'] = 0.0
    idx = sub_str.find('dropout')
    if idx != -1:
        idx2 = sub_str.find('_', idx)
        dropout_p = sub_str[idx + 7:idx2]
        dropout_p = dropout_p.replace('-', '.')
        f_args['dropout_p'] = float(dropout_p)

    # Check if augmentation
    f_args['data_augmentation'] = False
    idx = sub_str.find('augmentation')
    if idx != -1:
        f_args['data_augmentation'] = True

    # Check if split loss
    f_args['splitloss'] = False
    idx = sub_str.find('splitLoss')
    if idx != -1:
        f_args['splitloss'] = True

    # Check if symmetry
    f_args['closing_symmetry'] = False
    idx = sub_str.find('symmetry')
    if idx != -1:
        f_args['closing_symmetry'] = True

    # Check if flip closing angle
    f_args['randomly_flip_closing_angle'] = False
    idx = sub_str.find('rFlipClAng')
    if idx != -1:
        f_args['randomly_flip_closing_angle'] = True

    # Check if wrist centered
    f_args['center_to_wrist_joint'] = False
    idx = sub_str.find('wristCentered')
    if idx != -1:
        f_args['center_to_wrist_joint'] = True

    # Check if average pool is selected
    f_args['average_pool'] = False
    idx = sub_str.find('avgPool')
    if idx != -1:
        f_args['average_pool'] = True

    # Get the architecture type
    f_args['arch'] = Archs.PN
    idx = sub_str.find('arch')
    if idx != -1:
        idx2 = sub_str.find('_', idx)
        f_args['arch'] = int(sub_str[idx + 4: idx2])

    # Check if y axis is normalized
    f_args['yaxis_norm'] = False
    idx = sub_str.find('yAxisNorm')
    if idx != -1:
        f_args['yaxis_norm'] = True

    # Get the joint set
    f_args['joint_set'] = 1
    idx = sub_str.find('jointSet')
    if idx != -1:
        idx2 = sub_str.find('_', idx)
        f_args['joint_set'] = int(sub_str[idx + 8: idx2])

    return f_args


def load_regression_model(arch, k_out, dropout_p, average_pool, model=''):
    # Set up the model
    regressor = None
    if arch == Archs.PN:
        regressor = PointNetRegression(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Sym:
        regressor = PointNetRegressionSym(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_FC4:
        regressor = PointNetRegressionFC4(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_FC4_Sym:
        regressor = PointNetRegressionFC4Sym(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_FC45:
        regressor = PointNetRegressionFC45(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_FC45_Sym:
        regressor = PointNetRegressionFC45Sym(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Small_3L:
        regressor = PointNetRegressionSmall3Layers(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Small_4L:
        regressor = PointNetRegressionSmall4Layers(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Half:
        regressor = PointNetRegressionHalf(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Half_FC4:
        regressor = PointNetRegressionHalfFC4(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_FC4_256:
        regressor = PointNetRegressionFC4_256(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_LReLu:
        regressor = PointNetRegressionLeakyReLu(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Half_LReLu:
        regressor = PointNetRegressionHalfLeakyReLu(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Flat:
        regressor = PointNetRegressionFlat(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_NoPool:
        regressor = PointNetRegressionNoPool(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_NoPoolSmall:
        regressor = PointNetRegressionNoPoolSmall(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Flat5Layer:
        regressor = PointNetRegressionFlat5Layer(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    elif arch == Archs.PN_Flat_Split:
        regressor = PointNetRegressionFlatSplit(k_out=k_out, dropout_p=dropout_p, avg_pool=average_pool)
    else:
        print('Unknown architecture specified')
        return None

    if model != '':
        regressor.load_state_dict(torch.load(model))

    return regressor


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetfeat_custom(nn.Module):
    def __init__(self, avg_pool=False):
        super(PointNetfeat_custom, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.avg_pool = avg_pool

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x, None, None


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# PN = 1
class PointNetRegression(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegression, self).__init__()
        self.dropout_p = dropout_p
        self.feat = PointNetfeat_custom(avg_pool=avg_pool)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_Sym = 2
class PointNetRegressionSym(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionSym, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)
        self.bnf1 = nn.BatchNorm1d(256)
        self.bnf2 = nn.BatchNorm1d(512)
        self.bnf3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        # Generate feature
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_FC4 = 3
class PointNetRegressionFC4(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionFC4, self).__init__()
        self.dropout_p = dropout_p
        self.feat = PointNetfeat_custom(avg_pool=avg_pool)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        else:
            x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# PN_FC4_Sym = 4
class PointNetRegressionFC4Sym(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionFC4Sym, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)
        self.bnf1 = nn.BatchNorm1d(128)
        self.bnf2 = nn.BatchNorm1d(256)
        self.bnf3 = nn.BatchNorm1d(512)
        self.bnf4 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Generate feature
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = F.relu(self.bnf3(self.conv3(x)))
        x = self.bnf4(self.conv4(x))
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        else:
            x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# PN_FC45 = 5
class PointNetRegressionFC45(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionFC45, self).__init__()
        self.dropout_p = dropout_p
        self.feat = PointNetfeat_custom(avg_pool=avg_pool)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn4(self.dropout(self.fc4(x))))
        else:
            x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


# PN_FC45_Sym = 6
class PointNetRegressionFC45Sym(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionFC45Sym, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(128)
        self.bnf3 = nn.BatchNorm1d(256)
        self.bnf4 = nn.BatchNorm1d(512)
        self.bnf5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Generate feature
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = F.relu(self.bnf3(self.conv3(x)))
        x = F.relu(self.bnf4(self.conv4(x)))
        x = self.bnf5(self.conv5(x))
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn4(self.dropout(self.fc4(x))))
        else:
            x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


# PN_Small_3L = 7
class PointNetRegressionSmall3Layers(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionSmall3Layers, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(128)
        self.bnf3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Generate feature
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_Small_4L = 8
class PointNetRegressionSmall4Layers(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionSmall4Layers, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(128)
        self.bnf3 = nn.BatchNorm1d(256)
        self.bnf4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Generate feature
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = F.relu(self.bnf3(self.conv3(x)))
        x = self.bnf4(self.conv4(x))
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        else:
            x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# PN_Half = 9
class PointNetRegressionHalf(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionHalf, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 512, 1)
        self.bnf1 = nn.BatchNorm1d(32)
        self.bnf2 = nn.BatchNorm1d(64)
        self.bnf3 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Generate feature
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))  # Add ReLu here?
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_Half_FC4 = 10
class PointNetRegressionHalfFC4(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionHalfFC4, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 512, 1)
        self.bnf1 = nn.BatchNorm1d(32)
        self.bnf2 = nn.BatchNorm1d(64)
        self.bnf3 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Generate feature
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))  # Add ReLu here?
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        else:
            x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# PN_FC4_256 = 11
class PointNetRegressionFC4_256(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionFC4_256, self).__init__()
        self.dropout_p = dropout_p
        self.feat = PointNetfeat_custom(avg_pool=avg_pool)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        else:
            x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# PN_LReLu = 12
class PointNetRegressionLeakyReLu(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionLeakyReLu, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(128)
        self.bnf3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        # Generate feature
        x = F.leaky_relu(self.bnf1(self.conv1(x)))
        x = F.leaky_relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))  # Add ReLu here?
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected layers
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.leaky_relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_Half_LReLu = 13
class PointNetRegressionHalfLeakyReLu(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionHalfLeakyReLu, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 512, 1)
        self.bnf1 = nn.BatchNorm1d(32)
        self.bnf2 = nn.BatchNorm1d(64)
        self.bnf3 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Generate feature
        x = F.leaky_relu(self.bnf1(self.conv1(x)))
        x = F.leaky_relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))  # Add ReLu here?
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        # Fully connected layers
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.leaky_relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_Flat = 14
class PointNetRegressionFlat(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False, k_in=63):
        super(PointNetRegressionFlat, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(63, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(128)
        self.bnf3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bnf1(self.conv1(x.view(-1, 63, 1))))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))
        x = x.view(-1, 1024)

        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_NoPool = 15
class PointNetRegressionNoPool(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionNoPool, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(128)
        self.bnf3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(21504, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        num_points = x.size()[2]
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))
        x = x.view(-1, 1024 * num_points)

        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_NoPoolSmall = 16
class PointNetRegressionNoPoolSmall(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionNoPoolSmall, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(128)
        self.bnf3 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(2688, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        num_points = x.size()[2]
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))
        x = x.view(-1, 128 * num_points)

        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_Flat5Layer = 17
class PointNetRegressionFlat5Layer(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionFlat5Layer, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(63, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(64)
        self.bnf3 = nn.BatchNorm1d(64)
        self.bnf4 = nn.BatchNorm1d(128)
        self.bnf5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bnf1(self.conv1(x.view(-1, 63, 1))))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = F.relu(self.bnf3(self.conv3(x)))
        x = F.relu(self.bnf4(self.conv4(x)))
        x = self.bnf5(self.conv5(x))
        x = x.view(-1, 1024)

        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# PN_Flat_Split = 18
class PointNetRegressionFlatSplit(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegressionFlatSplit, self).__init__()
        self.dropout_p = dropout_p
        self.avg_pool = avg_pool

        self.conv1 = torch.nn.Conv1d(63, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bnf1 = nn.BatchNorm1d(64)
        self.bnf2 = nn.BatchNorm1d(128)
        self.bnf3 = nn.BatchNorm1d(1024)

        self.fc11 = nn.Linear(1024, 512)
        self.fc12 = nn.Linear(512, 256)
        self.fc13 = nn.Linear(256, 3)
        self.bn11 = nn.BatchNorm1d(512)
        self.bn12 = nn.BatchNorm1d(256)

        self.fc21 = nn.Linear(1024, 512)
        self.fc22 = nn.Linear(512, 256)
        self.fc23 = nn.Linear(256, 6)
        self.bn21 = nn.BatchNorm1d(512)
        self.bn22 = nn.BatchNorm1d(256)

        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = F.relu(self.bnf1(self.conv1(x.view(-1, 63, 1))))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = self.bnf3(self.conv3(x))
        x = x.view(-1, 1024)

        x1 = F.relu(self.bn11(self.fc11(x)))
        if self.dropout_p > 0.0:
            x1 = F.relu(self.bn12(self.dropout(self.fc12(x1))))
        else:
            x1 = F.relu(self.bn12(self.fc12(x1)))
        x1 = self.fc13(x1)

        x2 = F.relu(self.bn21(self.fc21(x)))
        if self.dropout_p > 0.0:
            x2 = F.relu(self.bn22(self.dropout(self.fc22(x2))))
        else:
            x2 = F.relu(self.bn22(self.fc22(x2)))
        x2 = self.fc23(x2)

        x = torch.cat((x1, x2), dim=1)
        return x


def compute_loss(prediction, target, offset, dist, points, loss_type=MSE_LOSS_CODE, independent_components=False,
                 lc_weights=[1./3., 1./3., 1./3.], closing_symmetry=True, reduction='mean', rot_as_mat=False,
                 yaxis_norm=False):
    if loss_type == MODEL_LOSS_CODE:
        return model_loss(prediction, target, offset, dist, points, closing_symmetry=closing_symmetry)
    else:
        if loss_type == L1_LOSS_CODE:
            return l1_loss(prediction, target, independent_components=independent_components, lc_weights=lc_weights,
                           closing_symmetry=closing_symmetry, reduction=reduction, rot_as_mat=rot_as_mat,
                           yaxis_norm=yaxis_norm)
        elif loss_type == SMOOTH_L1_LOSS_CODE:
            return smooth_l1_loss(prediction, target, independent_components=independent_components,
                                  lc_weights=lc_weights, closing_symmetry=closing_symmetry, reduction=reduction,
                                  rot_as_mat=rot_as_mat, yaxis_norm=yaxis_norm)
        else:
            return mse_loss(prediction, target, independent_components=independent_components, lc_weights=lc_weights,
                            closing_symmetry=closing_symmetry, reduction=reduction, rot_as_mat=rot_as_mat,
                            yaxis_norm=yaxis_norm)


def mse_loss(prediction, target, independent_components=False, lc_weights=[1./3., 1./3., 1./3.],
             closing_symmetry=True, reduction='mean', rot_as_mat=False, yaxis_norm=False):
    if rot_as_mat:
        return mse_loss_mat(prediction, target, lc_weights=[0.5, 0.5], closing_symmetry=closing_symmetry,
                            reduction=reduction)
    if independent_components:
        loss_translation = F.mse_loss(prediction[:, 0:3], target[:, 0:3], reduction='sum')
        loss_approach = F.mse_loss(prediction[:, 3:6], target[:, 3:6], reduction='sum')
        loss_closing = 0
        if not closing_symmetry:
            loss_closing = F.mse_loss(prediction[:, 6:9], target[:, 6:9], reduction='sum')
        else:
            for i in range(prediction.size()[0]):
                loss_closing += min(F.mse_loss(prediction[i, 6:9], target[i, 6:9], reduction='sum'),
                                    F.mse_loss(prediction[i, 6:9], -target[i, 6:9], reduction='sum'))

        if yaxis_norm:
            loss_translation /= prediction.size()[0]
            loss_approach /= prediction.size()[0]
            loss_closing /= prediction.size()[0]
        loss = lc_weights[0] * loss_translation + lc_weights[1] * loss_approach + lc_weights[2] * loss_closing
        if not yaxis_norm:
            denom = 3.0
            if reduction == 'mean':
                denom *= prediction.size()[0]
            loss = loss / denom
    else:
        loss = F.mse_loss(prediction, target, reduction=reduction)

    return loss


def mse_loss_mat(prediction, target, lc_weights=[0.5, 0.5],
                 closing_symmetry=True, reduction='mean'):
    loss_translation = F.mse_loss(prediction[:, 0:3], target[:, 0:3], reduction='sum')
    prediction_matrix = direction_to_matrix(prediction[:, 3:9])
    target_matrix = direction_to_matrix(target[:, 3:9])
    loss_orientation = 0
    if not closing_symmetry:
        loss_orientation = F.mse_loss(prediction_matrix, target_matrix, reduction='sum')
    else:
        for i in range(prediction.size()[0]):
            loss_orientation += min(F.mse_loss(prediction_matrix[i, :], target_matrix[i, :], reduction='sum'),
                                    F.mse_loss(prediction_matrix[i, :], -target_matrix[i, :], reduction='sum'))
    loss = lc_weights[0] * loss_translation + lc_weights[1] * loss_orientation
    denom = 2.0
    if reduction == 'mean':
        denom *= prediction.size()[0]
    loss = loss / denom

    return loss


def l1_loss(prediction, target, independent_components=False, lc_weights=[1./3., 1./3., 1./3.],
            closing_symmetry=True, reduction='mean', rot_as_mat=False, yaxis_norm=False):
    if rot_as_mat:
        return l1_loss_mat(prediction, target, lc_weights=[0.5, 0.5], closing_symmetry=closing_symmetry,
                           reduction=reduction)
    if independent_components:
        loss_translation = F.l1_loss(prediction[:, 0:3], target[:, 0:3], reduction='sum')
        loss_approach = F.l1_loss(prediction[:, 3:6], target[:, 3:6], reduction='sum')
        loss_closing = 0
        if not closing_symmetry:
            loss_closing = F.l1_loss(prediction[:, 6:9], target[:, 6:9], reduction='sum')
        else:
            for i in range(prediction.size()[0]):
                loss_closing += min(F.l1_loss(prediction[i, 6:9], target[i, 6:9], reduction='sum'),
                                    F.l1_loss(prediction[i, 6:9], -target[i, 6:9], reduction='sum'))

        if yaxis_norm:
            loss_translation /= prediction.size()[0]
            loss_approach /= prediction.size()[0]
            loss_closing /= prediction.size()[0]
        loss = lc_weights[0] * loss_translation + lc_weights[1] * loss_approach + lc_weights[2] * loss_closing
        if not yaxis_norm:
            denom = 3.0
            if reduction == 'mean':
                denom *= prediction.size()[0]
            loss = loss / denom
    else:
        loss = F.l1_loss(prediction, target, reduction=reduction)

    return loss


def l1_loss_mat(prediction, target, lc_weights=[0.5, 0.5],
                closing_symmetry=True, reduction='mean'):
    loss_translation = F.l1_loss(prediction[:, 0:3], target[:, 0:3], reduction='sum')
    prediction_matrix = direction_to_matrix(prediction[:, 3:9])
    target_matrix = direction_to_matrix(target[:, 3:9])
    loss_orientation = 0
    if not closing_symmetry:
        loss_orientation = F.l1_loss(prediction_matrix, target_matrix, reduction='sum')
    else:
        for i in range(prediction.size()[0]):
            loss_orientation += min(F.l1_loss(prediction_matrix[i, :], target_matrix[i, :], reduction='sum'),
                                    F.l1_loss(prediction_matrix[i, :], -target_matrix[i, :], reduction='sum'))
    loss = lc_weights[0] * loss_translation + lc_weights[1] * loss_orientation
    denom = 2.0
    if reduction == 'mean':
        denom *= prediction.size()[0]
    loss = loss / denom

    return loss


def smooth_l1_loss(prediction, target, independent_components=False, lc_weights=[1./3., 1./3., 1./3.],
                   closing_symmetry=True, reduction='mean', rot_as_mat=False, yaxis_norm=False):
    if rot_as_mat:
        return smooth_l1_loss_mat(prediction, target, lc_weights=[0.5, 0.5], closing_symmetry=closing_symmetry,
                                  reduction=reduction)
    if independent_components:
        loss_translation = F.smooth_l1_loss(prediction[:, 0:3], target[:, 0:3], reduction='sum')
        loss_approach = F.smooth_l1_loss(prediction[:, 3:6], target[:, 3:6], reduction='sum')
        loss_closing = 0
        if not closing_symmetry:
            loss_closing = F.smooth_l1_loss(prediction[:, 6:9], target[:, 6:9], reduction='sum')
        else:
            for i in range(prediction.size()[0]):
                loss_closing += min(F.smooth_l1_loss(prediction[i, 6:9], target[i, 6:9], reduction='sum'),
                                    F.smooth_l1_loss(prediction[i, 6:9], -target[i, 6:9], reduction='sum'))
        if yaxis_norm:
            loss_translation /= prediction.size()[0]
            loss_approach /= prediction.size()[0]
            loss_closing /= prediction.size()[0]
        loss = lc_weights[0] * loss_translation + lc_weights[1] * loss_approach + lc_weights[2] * loss_closing
        if not yaxis_norm:
            denom = 3.0
            if reduction == 'mean':
                denom *= prediction.size()[0]
            loss = loss / denom
    else:
        loss = F.smooth_l1_loss(prediction, target, reduction=reduction)

    return loss


def smooth_l1_loss_mat(prediction, target, lc_weights=[0.5, 0.5],
                       closing_symmetry=True, reduction='mean'):
    loss_translation = F.smooth_l1_loss(prediction[:, 0:3], target[:, 0:3], reduction='sum')
    prediction_matrix = direction_to_matrix(prediction[:, 3:9])
    target_matrix = direction_to_matrix(target[:, 3:9])
    loss_orientation = 0
    if not closing_symmetry:
        loss_orientation = F.smooth_l1_loss(prediction_matrix, target_matrix, reduction='sum')
    else:
        for i in range(prediction.size()[0]):
            loss_orientation += min(F.smooth_l1_loss(prediction_matrix[i, :], target_matrix[i, :], reduction='sum'),
                                    F.smooth_l1_loss(prediction_matrix[i, :], -target_matrix[i, :], reduction='sum'))
    loss = lc_weights[0] * loss_translation + lc_weights[1] * loss_orientation
    denom = 2.0
    if reduction == 'mean':
        denom *= prediction.size()[0]
    loss = loss / denom

    return loss


def direction_to_matrix(direction):
    direction_mat = np.zeros((direction.size()[0], 9))
    direction_np = direction.data.cpu().numpy()

    for i in range(direction.size()[0]):
        direction_mat[i, :] = error_def.to_rotation_matrix(direction_np[i, :]).reshape(1, 9)

    return torch.from_numpy(direction_np).float().cuda()


def model_loss(prediction, target, offset, dist, points, closing_symmetry=True):
    pred_np = prediction.data.cpu().numpy()
    targ_np = target.data.cpu().numpy()
    offset_np = offset.data.cpu().numpy().reshape((pred_np.shape[0], 3))
    dist_np = dist.data.cpu().numpy().reshape((pred_np.shape[0], 1))

    pred_tfs = error_def.to_transform_batch(pred_np, offset_np, dist_np)
    targ_tfs = error_def.to_transform_batch(targ_np, offset_np, dist_np)

    loss = 0
    for p, t in zip(pred_tfs, targ_tfs):
        if closing_symmetry:
            loss += error_def.add_symmetric_error(t, p, points)
        else:
            loss += error_def.add_error(t, p, points)

    # I could divide here by the size of the batch (pred_np.shape[0])

    loss = torch.tensor(loss, dtype=torch.float, requires_grad=True).cuda()

    return loss


def normalize_direction(tensor9):
    z_axis = tensor9[:, 3:6].clone()
    z_norm = torch.norm(z_axis, p=2, dim=1)
    z_axis = z_axis / z_norm[:, None]

    y_axis = tensor9[:, 6:9].clone()
    z_dot_y = torch.sum(z_axis * y_axis, dim=1)
    y_axis = y_axis - z_dot_y[:, None] * z_axis
    y_norm = torch.norm(y_axis, p=2, dim=1)
    y_axis = y_axis / y_norm[:, None]

    tensor9[:, 3:6] = z_axis
    tensor9[:, 6:9] = y_axis

    return tensor9


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
