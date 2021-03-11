import torch.nn as nn
import torch
import torch.nn.functional as F
from Net_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.4, 0.6], [64, 96], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(128, [0.6, 0.8], [96, 128], 256+256, [[128, 256, 512], [128, 256, 512]])
        self.fp3 = PointNetFeaturePropagation(in_channel=1024+512, mlp=[512, 256]) # 1536 = sa3中的特征1024 + sa2中的特征512(256+256)
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128]) # 576 = fp3中的特征256 + sa1中的特征320(64+128+128)
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 64]) # 134 = 128 + 3(l0_xyz的C) + 3(l0_points的C)
        self.conv1 = nn.Conv1d(64, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(32, 1, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.l1_loss(pred, target)

        return total_loss