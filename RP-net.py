#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import os

from comet_ml import Experiment
from transforms3d import euler
from scipy.spatial import ConvexHull

from dataset import Radardata
from torch.utils.data import DataLoader



import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import time
import sys
from net.refinement import Refinement
from net.llp_utils import get_mAP
from config import Config


def rotate_points(points, theta):
    '''theta is in radians
       points shape: (3,n)
    '''
    assert points.shape[0]==3
    return torch.matmul(torch.tensor(euler.euler2mat(0,0,0)).cuda().float(),points)


def get_bbox_corners(dim,params,theta=0):
    [d_x,d_y,d_z] = dim
    #rot = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rot = torch.tensor([[math.cos(theta), 0, math.sin(theta)], [0 ,1 ,0], [-math.sin(theta), 0, math.cos(theta)]]).cuda()
    points = [params[0:3] + torch.matmul(torch.tensor([-d_x/2, d_y/2, -d_z/2]).cuda(), rot), params[0:3] + torch.matmul(torch.tensor([-d_x/2, d_y/2, d_z/2]).cuda(),rot),
            params[0:3] + torch.matmul(torch.tensor([d_x/2, d_y/2, d_z/2]).cuda(), rot), params[0:3] + torch.matmul(torch.tensor([d_x/2, d_y/2, -d_z/2]).cuda(), rot),
            params[0:3] + torch.matmul(torch.tensor([-d_x/2, -d_y/2, -d_z/2]).cuda(), rot), params[0:3] + torch.matmul(torch.tensor([-d_x/2, -d_y/2, d_z/2]).cuda(),rot),
            params[0:3] + torch.matmul(torch.tensor([d_x/2, -d_y/2, d_z/2]).cuda(), rot), params[0:3] + torch.matmul(torch.tensor([d_x/2, -d_y/2, -d_z/2]).cuda(), rot)]
    # print("printttttttttttttttttt",points)
    return torch.stack(points)


def get_points_in_bbox(corners,pts): #, l, b, h, center, theta=0
    #todo: if want to see rotated bbox points rotate the points first..rn implemented for zero yaw angle
    # assert isinstance(pts, np.ndarray)


    mask_x = torch.cuda.ByteTensor(pts[:,0]<= max(corners[0,0], corners[2,0])) & torch.cuda.ByteTensor(pts[:,0]>= min(corners[0,0], corners[2,0]))
    mask_y = torch.cuda.ByteTensor(pts[:,1]<= max(corners[0,1], corners[4,1])) & torch.cuda.ByteTensor(pts[:,1]>= min(corners[0,1], corners[4,1]))
    mask_z = torch.cuda.ByteTensor(pts[:,2]<= max(corners[0,2], corners[2,2])) & torch.cuda.ByteTensor(pts[:,2]>= min(corners[0,2], corners[2,2]))

    # doubt in mask_z: the get_bbox code for corner points computation seems odd
    res_mask = mask_x*mask_y*mask_z
    #print(np.sum(res_mask), np.sum(mask_x), np.sum(mask_y), np.sum(mask_z))
    #print(pts.shape, mask_x.shape, mask_y.shape, mask_z.shape, res_mask.shape)
    return torch.t(pts[res_mask, :]), res_mask




def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = math.sqrt(torch.sum((corners[0,:] - corners[1,:])**2))
    b = math.sqrt(torch.sum((corners[1,:] - corners[2,:])**2))
    c = math.sqrt(torch.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    #print(rect1)
    # area1 = poly_area(torch.tensor(rect1,dtype=torch.float32)[:,0], torch.tensor(rect1,dtype=torch.float32)[:,1])
    # area2 = poly_area(torch.tensor(rect2,dtype=torch.float32)[:,0], torch.tensor(rect2,dtype=torch.float32)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
#     iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
#     print(inter_area)
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
#    print(vol2)
#    print(torch.div(inter_vol,(vol1+vol2-inter_vol)))
#    print(torch.div(inter_vol,(vol2)))
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou



def non_max_suppression(boxes, confidence_scores, nBox, overlapThresh=0.7):
   # if there are no boxes, return an empty list
    if len(boxes) == 0:
        print("no boxes input to nms calculation")
        return []
    # initialize the list of picked indexes
    pick = []

    idxs = torch.argsort(confidence_scores,0)

    # Sort by confidence interval ascending
    # print(len(idxs))
    # keep looping while some indexes still remain in the indexes
    # list
    while len(pick) < nBox:
       # grab the last index in the indexes list, add the index
       # value to the list of picked indexes, then initialize
       # the suppression list (i.e. indexes that will be deleted)
       # using the last index
       
        last = len(idxs) - 1
        i = idxs[last]
        # print("Confidences", confidence_scores[i])
        if confidence_scores[i] < 0.2:
           break
        pick.append(i.item())

        suppress = [last]
        # loop over all indexes in the indexes list
        corners_box = get_bbox_corners(boxes[last,:3],boxes[last,3:6],boxes[last,6])
        for pos in range(0, last):
           # grab the current index
            j = idxs[pos]
            try:
            # print(boxes.size())
                # print(pos)
                iou = box3d_iou(get_bbox_corners(boxes[pos,:3],boxes[pos,3:6],boxes[pos,6]), corners_box)
            except Exception as e:
                print(e)
                suppress.append(pos)
                continue
            # if there is sufficient overlap, suppress the
            # current bounding box
            if iou > overlapThresh:
               #print(pos)
                suppress.append(pos)
       # delete all indexes from the index list that are in the
       # suppression list
       # idxs = torch.delete(idxs, suppress)
        idxs[suppress] = 0
        idxs = idxs[torch.squeeze(torch.nonzero(idxs))]
        if len(idxs)==0:
           break
    # return only the bounding boxes that were picked
    # print("pick boxes",boxes,pick)
    if len(pick)==0:
        print("indices input.....................errorrrrrrrrrrr", idxs)
        pick = idxs[-nBox:]
    return boxes[pick], torch.tensor(pick)


if __name__ == '__main__':


    experiment = Experiment(api_key="",
                        project_name="", workspace="")

    #debt read ground truth labels
    classifier = Refinement(k=7)

    count = 0

    config = Config()
    
    experiment.log_asset("config.py")

    optimizer = optim.Adam(classifier.parameters(), lr=config.lr, weight_decay=0.00001)
    print("Total parameters:",sum(p.numel() for p in classifier.parameters()))
    print("Total trainable parameters:",sum(p.numel() for p in classifier.parameters() if p.requires_grad))


    radar_train_dataset = Radardata(256,False)
    radar_validation_dataset = Radardata(256,True)
    train_dataloader = DataLoader(radar_train_dataset, batch_size=config.batchsize, shuffle=True, drop_last=True, num_workers=config.workers)
    validation_dataloader = DataLoader(radar_validation_dataset, batch_size=config.batchsize_eval, shuffle=False, drop_last=True, num_workers=config.workers)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    
    # num_batches = int((files.shape[0])/config.batchsize)
    #print(num_batches)
    nepochs = config.nepochs
    num_classes = 2
    criterion = nn.CrossEntropyLoss(weight = torch.tensor([1,2],dtype=torch.float32).cuda())
    mse_loss = nn.MSELoss()
    smoothl1_loss = nn.SmoothL1Loss(reduction='mean')
    prev_recall = 0.0

    nPoints  = config.npoints
    total_loss_training = []
    total_rpn_loss_training = []
    total_loss_validation = []
    mAP_save = []


    for epoch in range(nepochs):
        train_acc_epoch, test_acc_epoch = [], []
        total_loss = 0.0
        total_loss_eval = 0.0
        total_rpn_loss = 0.0
        total_mIoU = 0.0
        total_mIoU_2d = 0.0
        epoch_timer = time.time()

        torch.cuda.empty_cache()
        print("start epoch")
        for batch, data in enumerate(train_dataloader):
            # print("Epoch", epoch)
            
            timer = time.time()
            
            points, lbls = data["points"], data["labels"]
            points_new = points[:,:,:config.nchannels]
            lbls = np.array(lbls)
            points_new = np.swapaxes(np.array(points_new), 1, 2) ##swapping y-z axis
            # points_new = np.array(points_new)

            #points= points[:,:,:100]
            points_new = torch.from_numpy(points_new).float()
            lbls = torch.from_numpy(lbls).float()

            # print(lbls.shape)
            points_new = points_new.to(device)#, lbls.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            classifier = classifier.train()
            


            x, confidence, select_anchor_box_batch_select, select_box_gt_label_idx_batch_select, pred_labels, labels_binary_iou, labels_binary_iou_select = classifier(points_new, lbls, True, epoch)

            
            pred_labels = pred_labels.view(-1,2)
            labels_binary_iou = labels_binary_iou.view(-1)
            rpn_loss = criterion(pred_labels+0.0000001, labels_binary_iou)# size changed to linearize

            rpn_loss += criterion(confidence+0.0000001, labels_binary_iou_select.view(-1))# size changed to linearize
            gt_labels_bbox = torch.stack([torch.index_select(lbls[ind,select_box_gt_label_idx_batch_select[ind,:],:], 1, torch.LongTensor([3,4,5,1,0,2,6]).cuda()) for ind in range(config.batchsize)])

            # print(gt_labels_bbox.size())

            if x.size()[0]!=0:
                # print(x,)
                center_x_loss = smoothl1_loss(labels_binary_iou_select.float()*x[:,:,0], labels_binary_iou_select.float()*(gt_labels_bbox-select_anchor_box_batch_select)[:,:,0]) #[x,y,z]
                center_y_loss = smoothl1_loss(labels_binary_iou_select.float()*x[:,:,1], labels_binary_iou_select.float()*(gt_labels_bbox-select_anchor_box_batch_select)[:,:,1]) #[x,y,z]
                center_z_loss = smoothl1_loss(labels_binary_iou_select.float()*x[:,:,2], labels_binary_iou_select.float()*(gt_labels_bbox-select_anchor_box_batch_select)[:,:,2]) #[x,y,z]
                angle_loss = smoothl1_loss(labels_binary_iou_select.float()*x[:,:,6], labels_binary_iou_select.float()*(gt_labels_bbox-select_anchor_box_batch_select)[:,:,6])
                h_loss = smoothl1_loss(labels_binary_iou_select.float()*x[:,:,3], labels_binary_iou_select.float()*(gt_labels_bbox-select_anchor_box_batch_select)[:,:,3]) #[h,w,l]
                w_loss = smoothl1_loss(labels_binary_iou_select.float()*x[:,:,4], labels_binary_iou_select.float()*(gt_labels_bbox-select_anchor_box_batch_select)[:,:,4]) #[h,w,l]
                l_loss = smoothl1_loss(labels_binary_iou_select.float()*x[:,:,5], labels_binary_iou_select.float()*(gt_labels_bbox-select_anchor_box_batch_select)[:,:,5]) #[h,w,l]
            
            # out = x+init_bbox
            loss = rpn_loss + angle_loss + center_x_loss + center_y_loss + center_z_loss + h_loss + w_loss + l_loss
            # print(angle_loss)
            total_rpn_loss += rpn_loss.item()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred_choice = pred_labels.data.max(1)[1]
            correct = pred_choice.eq(labels_binary_iou.data).cpu().sum()
            TP, FP,FN = 0, 0, 0
            for i in range(len(pred_choice)):
                if(pred_choice[i]==1 and labels_binary_iou.data[i]==1):
                    TP+=1
                elif(pred_choice[i]==0 and labels_binary_iou.data[i]==1):
                    FN+=1
                elif(pred_choice[i]==1 and labels_binary_iou.data[i]==0):
                    FP+=1
            train_acc = correct.item() / float(labels_binary_iou.size()[0])
            if(TP+FN!=0 and TP+FP!=0):
                precision = TP/(TP+FP)
                recall = TP/(TP+FN)
            else:
                precision = 0
                recall = 0
            s = 'epoch %d: /%d | train loss: %f | train acc: %f' % (epoch+1, batch+1, loss.item(), train_acc)
            train_acc_epoch.append(train_acc)

        print("................................................................total epoch time : %s seconds" %(time.time()-epoch_timer))
        print('Epoch %d training completed: total_loss: %f | total_rpn_loss: %f' % (epoch+1, total_loss/(batch+1),total_rpn_loss/(batch+1)))


        experiment.log_metrics({'Loss/train': total_loss/(batch+1)}, step=epoch+1)

        total_loss_training.append(total_loss)
        total_rpn_loss_training.append(total_rpn_loss)

        np.save('total_loss_training',total_loss_training)
        np.save('total_rpn_loss_training',total_rpn_loss_training)

        eval_timer = time.time()    
        torch.cuda.empty_cache()
        labels_boxes = []
        final_boxes_save = []
        confidence_save = []
        iou3d_save = []      
        gt_labels_save = []
        iou2d_save = []
        points_save = []
        nms_idx_save = []
        
        for batch, data in enumerate(validation_dataloader):
            
            points, lbls = data["points"], data["labels"]
            points_new = points[:,:,0:config.nchannels]
            lbls = np.array(lbls)
            points_new = np.swapaxes(np.array(points_new), 1, 2)
            # points_new = np.array(points_new)
            #points= points[:,:,:100]
            points_new = torch.from_numpy(points_new).float()
            lbls = torch.from_numpy(lbls).float()

            # print(lbls.shape)
            points_new = points_new.to(device)#, lbls.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            classifier = classifier.eval()

            
            x, confidence, select_anchor_box_batch_select, select_box_gt_label_idx_batch_select, pred_labels, labels_binary_iou, labels_binary_iou_select = classifier(points_new, lbls, False, epoch)
            # print("confidence: ",confidence)
            pred_labels = pred_labels.view(-1,2)
            labels_binary_iou = labels_binary_iou.view(-1)

            #Order is different from previous IoU calc
            residual = x[0].clone().detach()
            final_anchor_boxes = select_anchor_box_batch_select[0]+residual
            # print(residual[0])
            iou3d_eval, iou2d_eval = iou3d_utils.boxes_iou3d_gpu(final_anchor_boxes,torch.index_select(lbls[0], 1, torch.LongTensor([3,4,5,1,0,2,6]).cuda()))
            
            max_iou, max_iou_ind = torch.max(iou3d_eval,1)
            mIoU = torch.max(max_iou)
            max_iou_2d, max_iou_ind_2d = torch.max(iou2d_eval,1)
            mIoU_2d = torch.max(max_iou_2d)
            gt_labels_bbox = torch.stack([torch.index_select(lbls[ind,max_iou_ind_2d[:],:], 1, torch.LongTensor([3,4,5,1,0,2,6]).cuda()) for ind in range(config.batchsize_eval)])



            final_boxes_bev = kitti_utils.boxes3d_to_bev_torch_orig(final_anchor_boxes)
            final_idxs = iou3d_utils.nms_gpu(final_boxes_bev, confidence.view(6,2)[:,1], 0.1)

            if x.size()[0]!=0:
                # print(x,)
                center_x_loss = smoothl1_loss(x[:,:,0], (gt_labels_bbox-select_anchor_box_batch_select)[:,:,0]) #[x,y,z]
                center_y_loss = smoothl1_loss(x[:,:,1], (gt_labels_bbox-select_anchor_box_batch_select)[:,:,1]) #[x,y,z]
                center_z_loss = smoothl1_loss(x[:,:,2], (gt_labels_bbox-select_anchor_box_batch_select)[:,:,2]) #[x,y,z]
                angle_loss = smoothl1_loss(x[:,:,6], (gt_labels_bbox-select_anchor_box_batch_select)[:,:,6])
                h_loss = smoothl1_loss(x[:,:,3], (gt_labels_bbox-select_anchor_box_batch_select)[:,:,3]) #[h,w,l]
                w_loss = smoothl1_loss(x[:,:,4], (gt_labels_bbox-select_anchor_box_batch_select)[:,:,4]) #[h,w,l]
                l_loss = smoothl1_loss(x[:,:,5], (gt_labels_bbox-select_anchor_box_batch_select)[:,:,5]) #[h,w,l]
            
            loss = angle_loss + 5*center_x_loss + 5*center_y_loss + center_z_loss + h_loss + w_loss + l_loss
            total_loss_eval += loss.item()
            total_mIoU += mIoU
            total_mIoU_2d += mIoU_2d
            label_dim = 2
            labels_boxes.append(lbls.cpu().numpy().reshape(label_dim,7))
            # print(labels_boxes)
            
            # final_boxes_save.append(final_anchor_boxes.cpu().numpy().reshape(6,7))
            confidence_save.append(confidence.clone().detach().cpu().numpy().reshape(6,2))
            iou3d_save.append(iou3d_eval.cpu().numpy().reshape(6,label_dim))
            iou2d_save.append(iou2d_eval.cpu().numpy().reshape(6,label_dim))
            # gt_labels_save.append(gt_labels_bbox.cpu().numpy().reshape(6,7))
            # points_save.append(points.cpu().numpy().reshape(nPoints,-1))
            nms_idx_save.append(final_idxs.cpu().numpy())
        
        iou_2d_test = np.array(iou2d_save)
        max_iou_2d= np.amax(iou_2d_test,axis=1)
       
        total = 0
        iou_thresh = 0.5
        
        
            
        mAP,_,_ = get_mAP(np.array(confidence_save),np.array(iou2d_save),nms_idx_save,0.5)
        mAP_04,_,_ = get_mAP(np.array(confidence_save),np.array(iou2d_save),nms_idx_save,0.4)
        mAP_03,_,_ = get_mAP(np.array(confidence_save),np.array(iou2d_save),nms_idx_save,0.3)
        mAP_02,_,_ = get_mAP(np.array(confidence_save),np.array(iou2d_save),nms_idx_save,0.2)
        mAP_01,_,_ = get_mAP(np.array(confidence_save),np.array(iou2d_save),nms_idx_save,0.1)
        mAP_001, TP, mis_detect = get_mAP(np.array(confidence_save),np.array(iou2d_save),nms_idx_save,0.01)
        print('mAP:', mAP)
        print('0.4:', mAP_04, '0.3:',mAP_03,'0.2:',mAP_02, '0.1:',mAP_01,'0.01:',mAP_001)
        print('max mAP', prev_recall)


        experiment.log_metrics({'mAP/val': mAP}, step=epoch+1)
        experiment.log_metrics({'mAP_01/val': mAP_01}, step=epoch+1)
        experiment.log_metrics({'mAP_001/val': mAP_001}, step=epoch+1)
        experiment.log_metrics({'TP/val': TP}, step=epoch+1)
        experiment.log_metrics({'mis_detect/val': mis_detect}, step=epoch+1)
        experiment.log_metrics({'Loss/val': total_loss_eval/(batch+1)}, step=epoch+1)
        experiment.log_metrics({'mIoU_2d/val': total_mIoU_2d/(batch+1)}, step=epoch+1)


        if mAP>prev_recall:
            
            print("Epoch:",epoch, "Recall:",recall)
            prev_recall = mAP
        
        print('Epoch %d evaluation completed: total_loss: %f | mIoU: %f| mIoU_2d: %f | total_eval_time: %f'% (epoch+1, total_loss_eval/(batch+1),total_mIoU/(batch+1),total_mIoU_2d/(batch+1),time.time()-eval_timer))
        
        total_loss_validation.append(total_loss_eval)
        mAP_save.append(mAP)
        np.save('total_loss_validation',total_loss_validation)
        np.save('mAP_save',mAP_save)

        try:
            if epoch%2==1:
                torch.save(classifier, './models/epoch_'+str(epoch)+'.pth')
                np.save('./results/epoch_'+str(epoch)+'.npy',np.array(train_acc_epoch))
        except Exception as e:
            print(e)
            continue  

