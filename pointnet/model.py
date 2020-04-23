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


def compute_loss(prediction, target, offset, dist, points, loss_type=MSE_LOSS_CODE, independent_components=False,
                 lc_weights=[1. / 3., 1. / 3., 1. / 3.], closing_symmetry=True, reduction='mean', rot_as_mat=False):
    if loss_type == MODEL_LOSS_CODE:
        return model_loss(prediction, target, offset, dist, points, closing_symmetry=closing_symmetry)
    else:
        if loss_type == L1_LOSS_CODE:
            return l1_loss(prediction, target, independent_components=independent_components, lc_weights=lc_weights,
                           closing_symmetry=closing_symmetry, reduction=reduction, rot_as_mat=rot_as_mat)
        elif loss_type == SMOOTH_L1_LOSS_CODE:
            return smooth_l1_loss(prediction, target, independent_components=independent_components,
                                  lc_weights=lc_weights, closing_symmetry=closing_symmetry, reduction=reduction,
                                  rot_as_mat=rot_as_mat)
        else:
            return mse_loss(prediction, target, independent_components=independent_components, lc_weights=lc_weights,
                            closing_symmetry=closing_symmetry, reduction=reduction, rot_as_mat=rot_as_mat)


def mse_loss(prediction, target, independent_components=False, lc_weights=[1./3., 1./3., 1./3.],
             closing_symmetry=True, reduction='mean', rot_as_mat=False):
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
        loss = lc_weights[0] * loss_translation + lc_weights[0] * loss_approach + lc_weights[0] * loss_closing
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
    loss = lc_weights[0] * loss_translation + lc_weights[0] * loss_orientation
    denom = 2.0
    if reduction == 'mean':
        denom *= prediction.size()[0]
    loss = loss / denom

    return loss


def l1_loss(prediction, target, independent_components=False, lc_weights=[1./3., 1./3., 1./3.],
            closing_symmetry=True, reduction='mean', rot_as_mat=False):
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
        loss = lc_weights[0] * loss_translation + lc_weights[0] * loss_approach + lc_weights[0] * loss_closing
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
    loss = lc_weights[0] * loss_translation + lc_weights[0] * loss_orientation
    denom = 2.0
    if reduction == 'mean':
        denom *= prediction.size()[0]
    loss = loss / denom

    return loss


def smooth_l1_loss(prediction, target, independent_components=False, lc_weights=[1./3., 1./3., 1./3.],
                   closing_symmetry=True, reduction='mean', rot_as_mat=False):
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
        loss = lc_weights[0] * loss_translation + lc_weights[0] * loss_approach + lc_weights[0] * loss_closing
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
    loss = lc_weights[0] * loss_translation + lc_weights[0] * loss_orientation
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
