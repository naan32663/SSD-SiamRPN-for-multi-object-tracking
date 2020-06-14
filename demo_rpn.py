#!/usr/bin/python

import argparse
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from demo_rpn_utils.net import *
from demo_rpn_utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from demo_rpn_utils.utils import get_axis_aligned_bbox, cxy_wh_2_rect, load_net


parser = argparse.ArgumentParser(description='PyTorch SiameseX demo')

parser.add_argument('--model', metavar='model', default='SiamRPNPPRes50', type=str,
                    help='which model to use.')
#parser.add_argument('--model', metavar='model', default='SiamRPNResNeXt22', type=str,
#                    help='which model to use.')
args = parser.parse_args()

# load net
net = eval(args.model)()
load_net('./cp/{}.pth'.format(args.model), net)
net.eval().cuda()

print(net)


# image and init box
image_files = sorted(glob.glob('./data/test6/*.jpg'))
init_rbox = [522,1363,522,851,912,851,912,1363]

[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_files[0])  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net, args.model)

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
out_path="./data/output/" 

# tracking and visualization
toc = 0
for f, image_file in enumerate(image_files):
    im = cv2.imread(image_file)
    im = im[...,::-1]
    
    plt.figure(figsize=(20,12))
    fig = plt.gcf()
    plt.imshow(im)
    current_axis = plt.gca()
    plt.axis('off')
    
    print(im.shape)
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, im)  # track
    toc += cv2.getTickCount()-tic

    w = state['target_sz'][0]
    h = state['target_sz'][1]
    xmin = state['target_pos'][0] - w/2
    ymin = state['target_pos'][1] - h/2

 
    print(state['score'])
    idno = 896183
    color = colors[1]
    label = "ID : {:0>6d}".format(idno)
    current_axis.add_patch(plt.Rectangle((xmin, ymin), w, h, color=color, fill=False, linewidth=2))  
#    current_axis.text(cx, cy+h, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
        
    out_img_path =os.path.join(out_path,"{:0>4d}".format(f))
    fig.savefig(out_img_path, bbox_inches='tight', dpi=300, pad_inches=0.0)   

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
