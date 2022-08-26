# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:20:50 2019

@author: ksban
"""
import numpy as np
import pandas as pd
import glob
import json
import copy
import os

from pointillism import pointillism


def get_bbox_from_json(ind):
    global test_no
    labels_dir = "./data/scene"+str(test_no)
    labels_path = labels_dir + '/label/'+(6-len(str(ind)))*'0'+str(ind)+'.json'

    labels = []
    with open(labels_path) as f1:
        data2 = json.load(f1)
    
    b = data2['labels'][0]
    center = [b['center']['x'], b['center']['y'], b['center']['z']]
    d_x = b['size']['x']
    d_y = b['size']['y']
    d_z = b['size']['z']
    theta = -b['orientation']['z']

    labels.append(np.concatenate(([d_y, d_z, d_x], center, [theta]), axis=0))  # [w,h,l,x,y,z,theta]
    labels = np.array(labels)
    labels_padded = np.pad(labels, ((2 - labels.shape[0], 0), (0, 0)), 'edge')
    return labels_padded, labels


def filter_pts(pc, radius, confidence=None):
    temp = copy.deepcopy(pc)
    dist = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    temp = pc[dist > radius]
    if confidence is not None:
        cnf = confidence[dist < radius]
        return temp, cnf
    else:
        return temp


inputs = []
input_labels = []
input_labels_orig = []
test_nos = []
frame_nos = []

pntlsm = pointillism()
count = 0
count_rem = 0
val_ind = []

nPointsperFrame = 70
minPointsperFrame = 20

for test_no in range(13, 61):

    test_dir = "./data/scene"+str(test_no)
    radar_frames = glob.glob(test_dir + '/radar_0/*.csv')

    for i in range(len(radar_frames)):
        try:
            
            radar_0 = pd.read_csv(test_dir+"/radar_0/"+(6-len(str(i)))*'0'+str(i)+'.csv', delimiter=',')
            radar_1 = pd.read_csv(test_dir+"/radar_1/"+(6-len(str(i)))*'0'+str(i)+'.csv', delimiter=',')

        except Exception as e:
            print(e)
            continue

        radar_0 = np.array(radar_0.values[:, [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]])
        radar_1 = np.array(radar_1.values[:, [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]])

        # Offset compensation
        radar_0[:, 3] = radar_0[:, 3] - 0.51
        radar_1[:, 3] = radar_1[:, 3] + 0.51

        data_0 = radar_0[:, [2, 3, 4, 6, 9]]
        data_1 = radar_1[:, [2, 3, 4, 6, 9]]
        data_0[:, 2] = np.negative(radar_0[:, 4])
        data_1[:, 2] = np.negative(radar_1[:, 4])

        # Remove origin noise
        data_0 = filter_pts(data_0, 0.5)
        data_1 = filter_pts(data_1, 0.5)

        # remove noise points below ground
        data_0 = data_0[(data_0[:, 2] > -1) * (data_0[:, 2] < 2)]
        data_1 = data_1[(data_1[:, 2] > -1) * (data_1[:, 2] < 2)]

        ## add potentials
        eps = 0.5
        min_samples = 1
        [pot_0,pot_1] = pntlsm.find_llpc(data_0,data_1,eps,min_samples,True)
        data_0 = np.hstack((data_0,pot_0.reshape(-1,1),np.tile([1,0],(data_0.shape[0],1))))
        data_1 = np.hstack((data_1,pot_1.reshape(-1,1),np.tile([0,1],(data_1.shape[0],1))))

        data_combined = np.concatenate((data_0, data_1), axis=0)
        data_combined[:, 0] = -data_combined[:, 0]

        if data_combined.shape[0] >= nPointsperFrame:

            data_combined = data_combined[np.random.choice(data_combined.shape[0], nPointsperFrame, replace=False), :]
            try:
                labels, labels_orig = get_bbox_from_json(i)
            except Exception as e:
                print("Test_no:", test_no, "Frame_no:", i, "Excpetion:", e)
                continue


        elif data_combined.shape[0] >= minPointsperFrame:
            repeat_points = data_combined[
                            np.random.choice(data_combined.shape[0], nPointsperFrame - data_combined.shape[0],
                                             replace=True), :]
            data_combined = np.concatenate((data_combined, repeat_points), axis=0)
            try:
                labels, labels_orig = get_bbox_from_json(i)
            except Exception as e:
                print("Test_no:", test_no, "Frame_no:", i, "Excpetion:", e)
                continue

        else:
            count_rem += 1
            continue

        inputs.append(copy.deepcopy(data_combined))
        input_labels.append(labels)
        input_labels_orig.append(labels_orig)
        test_nos.append(test_no)
        frame_nos.append(i)

        if test_no in list([2, 16, 19, 20, 34, 36, 38, 41, 43, 50, 55, 58, 61]):
            val_ind.append(count)

        count += 1

inputs = np.array(inputs)
input_labels = np.array(input_labels)
test_nos = np.array(test_nos)
frame_nos = np.array(frame_nos)

indices = np.array(range(inputs.shape[0]))
np.random.shuffle(indices)

a = np.array([i for i in range(len(indices)) if i not in val_ind])
train_indices = indices[a]

if not os.path.exists("input_files"):
    os.mkdir("input_files")

## Save the processed data
np.save('input_files/test_nos', test_nos)
np.save('input_files/frame_nos', frame_nos)
np.save('input_files/inputs', inputs)
np.save('input_files/labels', input_labels)
np.save('input_files/val_indices', val_ind)
np.save('input_files/train_indices', train_indices)