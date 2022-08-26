import numpy as np 
from transforms3d import euler

from dataset import Radardata
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import time
from config import Config

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.config = Config()
        self.conv1 = nn.Conv1d(self.config.nchannels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_feat = global_feat


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x

class RPN(nn.Module):
    def __init__(self, k=2):
        super(RPN, self).__init__()
        self.config = Config()
        self.k = k
        self.feat = PointNetfeat(global_feat=True)
        self.centers = torch.tensor([[0,1,0],[0,-1,0],[2.5,-1,0], [2.5,1,0],[2.5,0,0]]).cuda()
        
        self.new_centers = torch.tensor([])
        self.theta = torch.tensor([[0]],dtype=torch.float32).cuda()
        for ind in range(1):
            rot = torch.tensor([[torch.cos(self.theta[ind]), 0, torch.sin(self.theta[ind])], [-torch.sin(self.theta[ind]), 0, torch.cos(self.theta[ind])], [0 ,1 ,0]]).cuda()
            if ind ==0:
                self.new_centers = torch.matmul(self.centers,rot)
            else:
                self.new_centers = torch.cat((self.new_centers,torch.matmul(self.centers,rot)))

        self.dim = torch.tensor([2,2,5],dtype=torch.float32).repeat(5*self.config.npoints,1).cuda()

        
        self.theta = torch.repeat_interleave(self.theta,5,0)
        self.theta = self.theta.repeat(self.config.npoints,1)

        self.conv0 = nn.Conv1d(131, 256, 1)
        self.bn0 = nn.BatchNorm1d(256)

        self.conv1 = nn.Conv1d(259, 512, 1)
        self.conv2 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.config = Config()

        
    def forward(self, points, gt_labels=[], is_training=True):
        x =  self.feat(points)
        b_size = points.size()[0]
        points = points[:,0:3,:].view(b_size,3,-1)
        iou_thresh_h = 0.5
        
        select_anchor_box_batch = torch.tensor([])
        select_box_gt_label_idx_batch = torch.tensor([])
        select_box_iou_batch = torch.tensor([])

        anchor_boxes_list = torch.tensor([])
        anchor_box_idx = 0
        
        for i in range(points.size()[0]):
            labels = gt_labels[i]
            anchor_box_idx +=1
                    
            pnts = points[i].permute(1,0)
            pnts_rep = torch.repeat_interleave(pnts,5,0)
            centers = pnts_rep - self.new_centers.repeat(self.config.npoints,1)
            theta_gt = torch.tensor([labels[0,-2]],dtype=torch.float32).cuda()
            theta_gt = theta_gt.repeat(5*self.config.npoints,1)

            theta_estimates = torch.tensor([labels[0,-1]],dtype=torch.float32).cuda()
            theta_estimates = theta_estimates.repeat(5*self.config.npoints,1)

            anchor_boxes_list = torch.cat((self.dim,centers,self.theta),1)

            if is_training:
                iou3d,iou2d = iou3d_utils.boxes_iou3d_gpu_orig(torch.index_select(labels, 1, torch.LongTensor([3,4,5,1,0,2,6]).cuda()), torch.index_select(anchor_boxes_list, 1, torch.LongTensor([3,4,5,1,0,2,6]).cuda()))
                max_iou, max_iou_ind = torch.max(iou2d,0)
                sort_iou_ind = torch.argsort(max_iou)

                n_boxes_max = self.config.n_boxes_max
                n_boxes_min = self.config.n_boxes_min
                select_anchor_box_single = torch.cat((anchor_boxes_list[sort_iou_ind[:n_boxes_min]],anchor_boxes_list[sort_iou_ind[-n_boxes_max:]]),0)
                select_anchor_box_single = torch.index_select(select_anchor_box_single, 1, torch.LongTensor([3,4,5,1,0,2,6]).cuda()).view(1,n_boxes_min+n_boxes_max,7)

                select_box_gt_label_idx_single = torch.cat((max_iou_ind[sort_iou_ind[:n_boxes_min]],max_iou_ind[sort_iou_ind[-n_boxes_max:]]),0).view(1,n_boxes_min+n_boxes_max)

                select_box_iou_single = torch.cat((max_iou[sort_iou_ind[:n_boxes_min]],max_iou[sort_iou_ind[-n_boxes_max:]]),0).view(1,n_boxes_min+n_boxes_max)
                
                if(select_anchor_box_batch.size()[0]==0):
                    select_anchor_box_batch = select_anchor_box_single
                    select_box_gt_label_idx_batch = select_box_gt_label_idx_single
                    select_box_iou_batch = select_box_iou_single
                else:
                    select_anchor_box_batch = torch.cat((select_anchor_box_batch,select_anchor_box_single),0)
                    select_box_gt_label_idx_batch = torch.cat((select_box_gt_label_idx_batch,select_box_gt_label_idx_single),0)
                    select_box_iou_batch = torch.cat((select_box_iou_batch,select_box_iou_single),0)
            

            else:
                select_anchor_box_single = torch.index_select(anchor_boxes_list, 1, torch.LongTensor([3,4,5,1,0,2,6]).cuda()).view(1,anchor_boxes_list.size()[0],7)

                
                select_box_gt_label_idx_single = torch.randn(1,anchor_boxes_list.size()[0]).view(1,anchor_boxes_list.size()[0])
                select_box_iou_single = torch.randn(1,anchor_boxes_list.size()[0]).view(1,anchor_boxes_list.size()[0])
                
                if(select_anchor_box_batch.size()[0]==0):
                    select_anchor_box_batch = select_anchor_box_single
                    select_box_gt_label_idx_batch = select_box_gt_label_idx_single
                    select_box_iou_batch = select_box_iou_single
                else:
                    select_anchor_box_batch = torch.cat((select_anchor_box_batch,select_anchor_box_single),0)
                    select_box_gt_label_idx_batch = torch.cat((select_box_gt_label_idx_batch,select_box_gt_label_idx_single),0)
                    select_box_iou_batch = torch.cat((select_box_iou_batch,select_box_iou_single),0)

        n_features_pool = 64
        pooled_features, pooled_empty_flag = roipool3d_utils.roipool3d_gpu(points.permute(0,2,1), x.permute(0,2,1), select_anchor_box_batch, 0.2,n_features_pool)
        subtract_center = select_anchor_box_batch.repeat(1,1,n_features_pool).view(select_anchor_box_batch.size()[0],select_anchor_box_batch.size()[1],n_features_pool,7)
        pooled_features[:,:,:,-3:] = pooled_features[:,:,:,-3:] - subtract_center[:,:,:,:3]


        pooled_features = pooled_features.view(-1,n_features_pool,259).permute(0,2,1)
        feats = self.bn1(self.conv1(pooled_features))
        feats = self.bn2(self.conv2(feats))
        if is_training:
            feats = feats.view(b_size,n_boxes_min+n_boxes_max,1024,n_features_pool)
        else:
            feats = feats.view(b_size,anchor_boxes_list.size()[0],1024,n_features_pool)
        
        feats = torch.max(feats, 3)[0]
        
        labels_binary_iou = select_box_iou_batch>iou_thresh_h

        labels_binary_iou = labels_binary_iou.long()
        save_boxes = select_anchor_box_batch.clone().detach().cpu()
        return feats.cuda(), labels_binary_iou.cuda(), select_box_iou_batch.cuda(), select_anchor_box_batch.cuda(), labels.cuda(), select_box_gt_label_idx_batch.cuda()
        

class PointNetCls(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls, self).__init__()
        self.k = k
        self.feat = RPN()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(512)    
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.sf = nn.Softmax(dim=1)

    def forward(self, x, gt_labels=[], is_training=True):
        features, labels_binary_iou, select_box_iou_batch, select_anchor_box_batch, labels, select_box_gt_label_idx_batch  = self.feat(x, gt_labels, is_training)
        x = F.relu(self.bn1(self.fc1(features.view(-1,1024))))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return features, self.sf(x), labels_binary_iou, select_box_iou_batch, select_anchor_box_batch, labels, select_box_gt_label_idx_batch