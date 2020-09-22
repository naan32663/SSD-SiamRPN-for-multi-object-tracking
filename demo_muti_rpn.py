# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 09:08:24 2020

@author: Anna
"""

#!/usr/bin/python

import argparse
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from ssd import build_ssd
from demo_rpn_utils.net import *
from demo_rpn_utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from demo_rpn_utils.utils import get_axis_aligned_bbox, cxy_wh_2_rect, load_net


parser = argparse.ArgumentParser(description='PyTorch SiameseX demo')

parser.add_argument('--img_path', default='images/', 
                    type=str, help='Dir to save results')
parser.add_argument('--save_folder', default='eval/', 
                    type=str, help='Dir to save results')
parser.add_argument('--ssd_model', default='VGG_coco_SSD_300x300_subsampled_7_classes.pth', 
                    type=str, help='Trained state_dict file path to open')
#parser.add_argument('--model', metavar='model', default='SiamRPNPPRes50', type=str, help='which model to use.')

parser.add_argument('--siam_model', metavar='model', default='SiamRPNResNeXt22', 
                    type=str, help='which model to use.')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

# load ssd
classes = ['background', 'pedestrian','bicycle','car','motorcycle','bus','truck','scooter']
num_classes = 7 + 1 # +1 background
detecnet = build_ssd('test', 300, num_classes) # initialize SSD
detecnet.load_state_dict(torch.load(args.trained_model))
detecnet.eval()
print('Finished loading SSD model!')

# load SiamRPN
tracknet = eval(args.siam_model)()
load_net('./cp/{}.pth'.format(args.siam_model), tracknet)
tracknet.eval()
print('Finished loading SiamRPN model!')

# load data
images = os.listdir(args.img_path)
mean = np.array((104, 117, 123), dtype=np.float32)
if args.cuda:
    detecnet = detecnet.cuda()
    tracknet = tracknet.cuda()
    cudnn.benchmark = True

tracker = Tracker(160, 5, 6, 100)

# evaluation
for img in images:
    img_name = str(img).split('.')[0]
    # Step 1: detection
    img = cv2.imread(os.path.join(args.img_path,img), cv2.IMREAD_COLOR)
    x = cv2.resize(img, (300, mean)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    x = torch.from_numpy(x.permute(2, 0, 1))
    x = Variable(x.unsqueeze(0))

    plt.figure(figsize=(20,12))
    fig = plt.gcf()
    plt.axis('off')
    plt.imshow(img)

    current_axis = plt.gca()
    
    if args.cuda:
        x = x.cuda()

    y = detecnet(x)      # forward pass
    detections = y.data
    
    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
    pred_num = 0
    targets = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = classes[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            target_pos, target_sz = np.array([pt[0], pt[1]]), np.array([pt[0]-pt[2], pt[1]-pt[3]])
            target = dict()
            target['target_pos'] = target_pos
            target['target_sz'] = target_sz
            target['target_img'] = img
            targets.append(target)
            pred_num += 1
            current_axis.add_patch(plt.Rectangle((pt[0], pt[1]), pt[0]-pt[2], pt[1]-pt[3], color=colors[i], fill=False, linewidth=2))  
            current_axis.text(pt[0], pt[1], label_name, size='small', bbox={'facecolor':colors[i], 'alpha':1.0})
            j += 1
    
    # Step 2: tracking    
    if (pred_num > 0):
        
        tracker.Update(targets, tracknet)
        # Visualize tracking results
        for i in range(len(tracker.tracks)):
            xmin = tracker.tracks[i].pos[0]
            ymax = tracker.tracks[i].pos[1]
            boxid = tracker.tracks[i].track_id
            label = "ID: {:0>6d}".format(boxid)
            current_axis.text(xmin, ymax, label, size='small', bbox={'facecolor':colors[i], 'alpha':1.0})

    out_img_path =os.path.join(args.save_folder,img_name+".jpg")
    fig.savefig(out_img_path, bbox_inches='tight', dpi=120, pad_inches=0.0)
