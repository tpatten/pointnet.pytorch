from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils.error_def as error_def


MSE_LOSS_CODE = 'MSE'
L1_LOSS_CODE = 'L1'
MODEL_LOSS_CODE = 'MODEL'


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
    def __init__(self, global_feat=True, avg_pool=False):
        super(PointNetfeat_custom, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.avg_pool = avg_pool

    def forward(self, x):
        n_pts = x.size()[2]
        trans = None
        x = F.relu(self.bn1(self.conv1(x)))
        trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.avg_pool:
            x = torch.mean(x, 2, keepdim=True)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


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


class PointNetRegression(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegression, self).__init__()
        self.dropout_p = dropout_p
        self.feat = PointNetfeat_custom(global_feat=True, avg_pool=avg_pool)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        if self.dropout_p > 0.0:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class PointNetRegressionFC4(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegression, self).__init__()
        self.dropout_p = dropout_p
        self.feat = PointNetfeat_custom(global_feat=True, avg_pool=avg_pool)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, k_out)
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

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


class PointNetRegressionFC45(nn.Module):
    def __init__(self, k_out=9, dropout_p=0.0, avg_pool=False):
        super(PointNetRegression, self).__init__()
        self.dropout_p = dropout_p
        self.feat = PointNetfeat_custom(global_feat=True, avg_pool=avg_pool)
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
        self.relu = nn.ReLU()

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


def compute_loss(prediction, target, offset, dist, points, loss_type=MSE_LOSS_CODE, independent_components=False,
                 lc_weights=[1. / 3., 1. / 3., 1. / 3.], closing_symmetry=True, reduction='mean'):
    if loss_type == MODEL_LOSS_CODE:
        return model_loss(prediction, target, offset, dist, points, closing_symmetry=closing_symmetry)
    else:
        if loss_type == L1_LOSS_CODE:
            return l1_loss(prediction, target, independent_components=independent_components, lc_weights=lc_weights,
                           closing_symmetry=closing_symmetry, reduction=reduction)
        else:
            return mse_loss(prediction, target, independent_components=independent_components, lc_weights=lc_weights,
                            closing_symmetry=closing_symmetry, reduction=reduction)


def mse_loss(prediction, target, independent_components=False, lc_weights=[1./3., 1./3., 1./3.],
             closing_symmetry=True, reduction='mean'):
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


def l1_loss(prediction, target, independent_components=False, lc_weights=[1./3., 1./3., 1./3.],
            closing_symmetry=True, reduction='mean'):
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
