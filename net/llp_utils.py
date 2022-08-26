import numpy as np

class box:
    def __init__(self,confidence,positivity):
        self.confidence = confidence
        self.positivity = positivity

def get_mAP(confidence,iou_2d,nms_idx,iou_thresh):
    det_boxes = []
    n_tests = confidence.shape[0]
    
    n_lables = 0
    for n in range(n_tests):
        n_lables+=len(np.unique(iou_2d[n,0,:]))
    for i in range(confidence.shape[0]):
        conf = confidence[i,nms_idx[i],1]
        iou = iou_2d[i,nms_idx[i],:]
        iou_max = np.max(iou,axis=1)
        sort_ind = np.argsort(iou_max)  
        sort_ind = sort_ind[::-1]
        conf= conf[sort_ind]
        iou = iou[sort_ind,:]
        unique,unique_ind = np.unique(iou[0,:],return_index=True)
        for index in range(len(nms_idx[i])):
            curr_conf = conf[index]
            if(len(unique_ind)>0):
                
                iou_max = np.max(iou[index,unique_ind])
                ind = np.argmax(iou[index,unique_ind])
                unique_ind = np.delete(unique_ind,ind)
                if iou_max>=iou_thresh:
                    det_boxes.append(box(curr_conf,'TP'))
                else:
                    det_boxes.append(box(curr_conf,'FP'))
            else:
                det_boxes.append(box(curr_conf,'FP'))
        
    
    det_boxes_sorted = sorted(det_boxes, key=lambda k: k.confidence)
    det_boxes_sorted = det_boxes_sorted[::-1]    
    
    pr_precision=[]
    pr_recall = []
    TP = 0
    FP = 0
    
    for r,curr_box in enumerate(det_boxes_sorted):
        if curr_box.positivity=='TP':
            TP+=1
        pr_precision.append(TP/(r+1))
        pr_recall.append(TP/n_lables)
    
    for r in range(len(pr_precision)-1):
        pr_precision[r]=np.max(pr_precision[r:])

  
    area = 0
    for r in range(len(pr_precision)-1):
        area+=(pr_recall[r+1]-pr_recall[r])*pr_precision[r]
        
    return area, TP, (1- TP/n_lables)