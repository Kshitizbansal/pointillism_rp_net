from transforms3d import euler
from dataset import Radardata
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import time
from net.RPN import RPN
from config import Config

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

class Refinement(nn.Module):
    def __init__(self, k=7):
        super(Refinement, self).__init__()
        self.k = k
        self.feat = PointNetCls()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.fc4 = nn.Linear(256, 2)

        self.bn1 = nn.BatchNorm1d(512)    
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.sf = nn.Softmax(dim=1)
        self.config = Config()

    def forward(self, x, gt_labels=[], is_training=True, epoch_n=0):
        
        features, pred_labels, labels_binary_iou, select_box_iou_batch, select_anchor_box_batch, labels, select_box_gt_label_idx_batch = self.feat(x, gt_labels, is_training)
        timer = time.time()
        total_box = self.config.n_boxes_max+self.config.n_boxes_min
        b_size = select_box_iou_batch.size()[0]
        if is_training:
            pred_labels = pred_labels.view(b_size,total_box,2)
        else:
            pred_labels = pred_labels.view(b_size,select_anchor_box_batch.size()[1],2)
        
        temp = pred_labels.clone().detach()
        nms_idxs = torch.tensor([])
        nms_box_count = 6
        nms_thresh = 0.5
    
        for k in range(b_size):
            boxes_bev = kitti_utils.boxes3d_to_bev_torch_orig(select_anchor_box_batch[k])
            if epoch_n>=self.config.rpn_epochs or (not is_training):                                
                idxs = iou3d_utils.nms_gpu(boxes_bev, temp[k,:,1], nms_thresh)
            else:
                idxs = iou3d_utils.nms_gpu(boxes_bev, select_box_iou_batch[k], nms_thresh)

            if len(idxs)==0:
                print('indices:0')
                idxs = torch.tensor([total_box-6,total_box-5,total_box-4,total_box-3,total_box-2,total_box-1]).cuda()

            idxs = idxs[:nms_box_count] 
            while len(idxs)<nms_box_count:
                idxs = torch.cat((idxs,idxs[:nms_box_count-len(idxs)]))

            if k==0:
                nms_idxs = idxs.view(1,nms_box_count)
            else:
                nms_idxs = torch.cat((nms_idxs,idxs.view(1,nms_box_count)),0)
                    
     
        features_select = torch.stack([features[ind,nms_idxs[ind,:]] for ind in range(b_size)])
        select_anchor_box_batch_select = torch.stack([select_anchor_box_batch[ind,nms_idxs[ind,:]] for ind in range(b_size)])
        select_box_gt_label_idx_batch_select = torch.stack([select_box_gt_label_idx_batch[ind,nms_idxs[ind,:]] for ind in range(b_size)])
        labels_binary_iou_select = torch.stack([labels_binary_iou[ind,nms_idxs[ind,:]] for ind in range(b_size)])

        if len(idxs)==1:
            features_select = torch.cat((features_select,features_select),0)
            select_anchor_box_batch_select = torch.cat((select_anchor_box_batch_select,select_anchor_box_batch_select),0)
            select_box_gt_label_idx_batch_select = torch.cat((select_box_gt_label_idx_batch_select,select_box_gt_label_idx_batch_select),0)
        x = F.relu(self.bn1(self.fc1(features_select.view(-1,1024))))
        x = F.relu(self.bn2(self.fc2(x)))
        confidence = self.fc4(x)
        x = self.fc3(x)

        x = x.view(b_size,nms_box_count,-1)
        return x, self.sf(confidence), select_anchor_box_batch_select, select_box_gt_label_idx_batch_select, pred_labels, labels_binary_iou, labels_binary_iou_select