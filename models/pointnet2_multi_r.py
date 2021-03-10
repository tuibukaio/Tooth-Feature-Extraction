import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

'''
Dr.Wei's teeth feature points extract code (pytorch version [not complete yet])
'''

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(get_model, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        
        self.normal_channel = normal_channel

        self.sa01 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa02 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa03 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp03 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp02 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp01 = PointNetFeaturePropagation(in_channel=128+6+additional_channel, mlp=[128, 128, 128])

        self.sa04 = PointNetSetAbstraction(npoint=512, radius=0.3, nsample=64, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa05 = PointNetSetAbstraction(npoint=128, radius=0.5, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa06 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp04 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp05 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp06 = PointNetFeaturePropagation(in_channel=128+6+additional_channel, mlp=[128, 128, 128])

        self.conv00 = nn.Conv1d(128, 128, 1)
        self.conv01 = nn.Conv1d(128, 128, 1)
        
        self.drop00 = nn.Dropout(0.5)
        self.drop01 = nn.Dropout(0.5)

        # heatmap1
        self.sa11 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=256+3, mlp=[64, 64, 128], group_all=False)
        self.sa12 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa13 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.fp13 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp12 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp11 = PointNetFeaturePropagation(in_channel=387, mlp=[128, 128, 128])

        self.conv10 = nn.Conv1d(128, 512, 1)
        self.drop10 = nn.Dropout(0.5)
        self.conv11 = nn.Conv1d(512+3, 128, 1)
        self.drop11 = nn.Dropout(0.5)
        self.conv12 = nn.Conv1d(128+3, 32, 1)
        self.drop12 = nn.Dropout(0.5)
        self.conv13 = nn.Conv1d(32+3, 1, 1)
        self.drop13 = nn.Dropout(0.5)
        self.conv14 = nn.Conv1d(1, 1, 1)

        # heatmap2
        self.sa21 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=256+3, mlp=[64, 64, 128], group_all=False)
        self.sa22 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa23 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.fp23 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp22 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp21 = PointNetFeaturePropagation(in_channel=387, mlp=[128, 128, 128])

        self.conv20 = nn.Conv1d(128, 512, 1)
        self.drop20 = nn.Dropout(0.5)
        self.conv21 = nn.Conv1d(512+3, 128, 1)
        self.drop21 = nn.Dropout(0.5)
        self.conv22 = nn.Conv1d(128+3, 32, 1)
        self.drop22 = nn.Dropout(0.5)
        self.conv23 = nn.Conv1d(32+3, 1, 1)
        self.drop23 = nn.Dropout(0.5)
        self.conv24 = nn.Conv1d(1, 1, 1)

        # heatmap3
        self.sa31 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=256+3, mlp=[64, 64, 128], group_all=False)
        self.sa32 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa33 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.fp33 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp32 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp31 = PointNetFeaturePropagation(in_channel=387, mlp=[128, 128, 128])

        self.conv30 = nn.Conv1d(128, 512, 1)
        self.drop30 = nn.Dropout(0.5)
        self.conv31 = nn.Conv1d(512+3, 128, 1)
        self.drop31 = nn.Dropout(0.5)
        self.conv32 = nn.Conv1d(128+3, 32, 1)
        self.drop32 = nn.Dropout(0.5)
        self.conv33 = nn.Conv1d(32+3, 1, 1)
        self.drop33 = nn.Dropout(0.5)
        self.conv34 = nn.Conv1d(1, 1, 1)

        # heatmap4
        self.sa41 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=256+3, mlp=[64, 64, 128], group_all=False)
        self.sa42 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa43 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.fp43 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp42 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp41 = PointNetFeaturePropagation(in_channel=387, mlp=[128, 128, 128])

        self.conv40 = nn.Conv1d(128, 512, 1)
        self.drop40 = nn.Dropout(0.5)
        self.conv41 = nn.Conv1d(512+3, 128, 1)
        self.drop41 = nn.Dropout(0.5)
        self.conv42 = nn.Conv1d(128+3, 32, 1)
        self.drop42 = nn.Dropout(0.5)
        self.conv43 = nn.Conv1d(32+3, 1, 1)
        self.drop43 = nn.Dropout(0.5)
        self.conv44 = nn.Conv1d(1, 1, 1)



    def forward(self,pointcloud):
        B,C,N = pointcloud.size()
        if self.normal_channel:
            l0_points = xyz[:, :, :]
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz[:, :, :]
            l0_xyz = xyz[:, :3, :]
        
        # set abstraction layers0
        l1_xyz0, l1_points0 = self.sa01(l0_xyz, l0_points)
        l2_xyz0, l2_points0 = self.sa02(l1_xyz0, l1_points0)
        l3_xyz0, l3_points0 = self.sa03(l2_xyz0, l2_points0)

        # feature propagation layers0
        l2_points0 = self.fp03(l2_xyz0, l3_xyz0, l2_points0, l3_points0)
        l1_points0 = self.fp02(l1_xyz0, l2_xyz0, l1_points0, l2_points0)
        l0_points0 = self.fp01(l0_xyz, l1_xyz0, torch.cat([l0_xyz, l0_points],1), l1_points0)

        # set abstraction layers1
        l1_xyz1, l1_points1 = self.sa04(l0_xyz, l0_points)
        l2_xyz1, l2_points1 = self.sa05(l1_xyz1, l1_points1)
        l3_xyz1, l3_points1 = self.sa06(l2_xyz1, l2_points1)

        # feature propagation layers1
        l2_points1 = self.fp04(l2_xyz1, l3_xyz1, l2_points1, l3_points1)
        l1_points1 = self.fp05(l1_xyz1, l2_xyz1, l1_points1, l2_points1)
        l0_points1 = self.fp06(l0_xyz, l1_xyz1, torch.cat([l0_xyz, l0_points],1), l1_points1)

        # FC layers
        net0 = self.conv00(l0_points0)
        net1 = self.conv01(l0_points1)
        
        net0 = self.drop00(net0)
        net1 = self.drop01(net1)

        net = torch.cat([net0,net1],1)

        l0_points1 = net
        l0_points2 = net
        l0_points3 = net
        l0_points4 = net

        # heatmap1
        l1_xyz, l1_points = self.sa11(l0_xyz, l0_points1)
        l2_xyz, l2_points = self.sa12(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa13(l2_xyz, l2_points)

        l2_points = self.fp13(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp12(l1_xyz, l2_xyz, l1_points, l2_points0)
        l0_points = self.fp11(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points1],1), l1_points)

        net = self.conv10(l0_points)
        net = self.drop10(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv11(net)
        net = self.drop11(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv12(net)
        net = self.drop12(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv13(net)
        net = self.drop13(net)
        heatmap1 = self.conv14(net)

        # heatmap2
        l1_xyz, l1_points = self.sa21(l0_xyz, l0_points2)
        l2_xyz, l2_points = self.sa22(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa23(l2_xyz, l2_points)

        l2_points = self.fp23(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp22(l1_xyz, l2_xyz, l1_points, l2_points0)
        l0_points = self.fp21(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points2],1), l1_points)

        net = self.conv20(l0_points)
        net = self.drop20(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv21(net)
        net = self.drop21(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv22(net)
        net = self.drop22(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv23(net)
        net = self.drop23(net)
        heatmap2 = self.conv24(net)

        # heatmap3
        l1_xyz, l1_points = self.sa31(l0_xyz, l0_points3)
        l2_xyz, l2_points = self.sa32(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa33(l2_xyz, l2_points)

        l2_points = self.fp33(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp32(l1_xyz, l2_xyz, l1_points, l2_points0)
        l0_points = self.fp31(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points3],1), l1_points)

        net = self.conv30(l0_points)
        net = self.drop30(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv31(net)
        net = self.drop31(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv32(net)
        net = self.drop32(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv33(net)
        net = self.drop33(net)
        heatmap3 = self.conv34(net)

        # heatmap4
        l1_xyz, l1_points = self.sa41(l0_xyz, l0_points4)
        l2_xyz, l2_points = self.sa42(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa43(l2_xyz, l2_points)

        l2_points = self.fp43(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp42(l1_xyz, l2_xyz, l1_points, l2_points0)
        l0_points = self.fp41(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points4],1), l1_points)

        net = self.conv40(l0_points)
        net = self.drop40(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv41(net)
        net = self.drop41(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv42(net)
        net = self.drop42(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv43(net)
        net = self.drop43(net)
        heatmap4 = self.conv44(net)

        return heatmap1, heatmap2, heatmap3, heatmap4

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, heatmap1, heatmap2, heatmap3, heatmap4, heatmap1_gt, heatmap2_gt, heatmap3_gt, heatmap4_gt):
        lossH1 = F.mse_loss(heatmap1, heatmap1_gt)
        lossH2 = F.mse_loss(heatmap2, heatmap2_gt)
        lossH3 = F.mse_loss(heatmap3, heatmap3_gt)
        lossH4 = F.mse_loss(heatmap4, heatmap4_gt)

        return lossH1+lossH2+lossH3+lossH4

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 3, 2048)
    h1,h2,h3,h4 = model(xyz)
    print(xyz.size())