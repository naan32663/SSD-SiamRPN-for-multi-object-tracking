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
import json

from demo_rpn_utils.net import *
from demo_rpn_utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from demo_rpn_utils.utils import get_axis_aligned_bbox, cxy_wh_2_rect, load_net


parser = argparse.ArgumentParser(description='PyTorch SiameseX demo')

#parser.add_argument('--model', metavar='model', default='SiamRPNPPRes50', type=str, help='which model to use.')

parser.add_argument('--model', metavar='model', default='SiamRPNResNeXt22', type=str, help='which model to use.')

args = parser.parse_args()

# load net
net = eval(args.model)()
load_net('./cp/{}.pth'.format(args.model), net)
net.eval().cuda()

#Read ith frame to init box
initbox=[]
with open("./data/test-orig/result.txt", "r") as f: 
#with open("./InputFrames/result.txt", "r") as f: 
    # read the contents of file line by line: 
    for line in f: 
     # split current line by comma to get array of items from it 
     array_parts = line.split(";") 
     # filter elements, removing empty, and stripping spaces: 
     array_parts = [item.strip() for item in array_parts if item.strip() != ""] 
     # add line's items into result array: 
     initbox.append(array_parts)   
# Printing result: 
#print(len(initbox[0])) 


#read images 
image_files = sorted(glob.glob('./data/test-orig/*.jpg'))
#image_files = sorted(glob.glob('./InputFrames/*.jpg'))

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
out_path="./output/" 
orgim_w = 1080
orgim_h = 1920

toc = 0
img_index=1
classes = ['background', 'pedestrian','bicycle','car','motorcycle','bus','truck','scooter']

#template list
templist=[]

# tracker init 
imtemp = cv2.imread(image_files[0]) 
#print (imtemp.shape)
imtemp_w = imtemp.shape[1]
imtemp_h = imtemp.shape[0]
 
for i in range(len(initbox[0])-1):
    box = initbox[0][i+1]
#    print(box)
    boxinfo = box.split(",")
    boxinfo = [item.strip() for item in boxinfo if item.strip() != ""]
#    print(boxinfo[2])
    classname = boxinfo[0]
    xmin = float(boxinfo[2])*imtemp_w/orgim_w
    ymin = float(boxinfo[3])*imtemp_h/orgim_h 
    w = float(boxinfo[4])*imtemp_w/orgim_w
    h = float(boxinfo[5])*imtemp_h/orgim_h        
    target_pos, target_sz = np.array([xmin, ymin]), np.array([w, h]) 
    
    #save th original image position for json
    org_pos, org_sz = np.array([float(boxinfo[2]),float(boxinfo[3])]), np.array([float(boxinfo[4]), float(boxinfo[5])]) 
    
    boxid = random.randint(100000,999999)
    boxes = dict()
    boxes['target_pos'] = target_pos
    boxes['target_sz'] = target_sz
    boxes['boxid'] = boxid
    boxes['count'] = 0
    boxes['classname'] = classname

#use this when save result to json    
    boxes['org_pos'] = org_pos
    boxes['org_sz'] = org_sz
    templist.append(boxes)
#    print(templist)

# save the tracking result to json         
jf = open("hypotheses.json","w")
frames = []

# tracking and visualization
for f, image_file in enumerate(image_files):
    
    img_name = "{:0>4d}".format(img_index) +'.jpg'
    img_index = img_index+1
#    print("Image name : " + img_name)

    
    # process 1th frame      
    if (f==0) :

        imdetec = cv2.imread(image_files[0])
#        plt.figure(figsize=(20,12))
#        fig = plt.gcf()
#        plt.imshow(imdetec)
#        current_axis = plt.gca()
#        plt.axis('off')
        hypotheses = []  #use this when you save the result to json
        
        print("In frame 0, templist len = ", len(templist))
        for i in range(len(templist)):
            boxes = templist[i]
#            print(boxes)
            boxid = boxes['boxid']
            color = colors[i]
            w = boxes['target_sz'][0]#.astype(np.float)
            h = boxes['target_sz'][1]
            xmin = boxes['target_pos'][0]
            ymin = boxes['target_pos'][1]
 
                       
            classname = classes[round(float(boxes['classname']))]

#            label = "ID : {:0>6d}".format(boxid)  
#            current_axis.text(xmin, ymin+h, label, size='medium', color='white', bbox={'facecolor':color, 'alpha':1.0})  

#save to json
            xmin_org = boxes['org_pos'][0]
            ymin_org = boxes['org_pos'][1]
            w_org = boxes['org_sz'][0]
            h_org = boxes['org_sz'][1]
            
            hypo = {
                'id':     classname,
                'x':      round(xmin_org,2),
                'y':      round(ymin_org,2),
                'width':  round(w_org,2),
                'height': round(h_org,2)                  
            }
            hypotheses.append(hypo)
        frameitem = {
            # no num
            'class':      'frame',
            'timestamp':  int(img_name.split('.')[0]),
            'hypotheses': hypotheses
        }
        frames.append(frameitem)
        print("frame len = "+ str(len(frames)))

#        out_img_path =os.path.join(out_path,img_name)
#        fig.savefig(out_img_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
        
   #process nth frames      
    else :
#        break
        impre = cv2.imread(image_files[f-1])
        imdetec = cv2.imread(image_files[f])
        hypotheses = []  #use this when you save the result to json
        
        # Determine if new targets appear
        for i in range(len(initbox[f])-1):
            box = initbox[f][i+1]
#            print(box)
            boxinfo = box.split(",")
            boxinfo = [item.strip() for item in boxinfo if item.strip() != ""]
#            print(boxinfo[2])
            xmin = float(boxinfo[2])*imtemp_w/orgim_w
            ymin = float(boxinfo[3])*imtemp_h/orgim_h  
            w = float(boxinfo[4])*imtemp_w/orgim_w
            h = float(boxinfo[5])*imtemp_h/orgim_h        
            target_pos, target_sz = np.array([xmin, ymin]), np.array([w, h])
            state = SiamRPN_init(imdetec, target_pos, target_sz, net, args.model)
            state = SiamRPN_track(state, impre)  # track
            
            
#            print("in preim state score = ", state['score'])
            if state['score'] < 0.7: # if similarity < 0.75 is a new target
                boxid = random.randint(100000,999999)# use this when draw the picture for demovideo
                org_pos, org_sz = np.array([float(boxinfo[2]), float(boxinfo[3])]), np.array([float(boxinfo[4]), float(boxinfo[5])])

                boxes = dict()
                boxes['target_pos'] = target_pos
                boxes['target_sz'] = target_sz
                boxes['boxid'] = boxid
                boxes['count'] = 0
                boxes['org_pos'] = org_pos
                boxes['org_sz'] = org_sz
                templist.append(boxes)

        # get the detec image
#        plt.figure(figsize=(20,12))
#        fig = plt.gcf()
#        plt.imshow(imdetec)
#        current_axis = plt.gca()
#        plt.axis('off')
        
        print("In frame %d  templist = %d" % (f, len(templist)))
        # tracking                
        for i in range(len(templist)):
#            print (i)
            boxes = templist[i]
#            print(boxes)
            boxid = boxes['boxid']
            count = boxes['count']
            color = colors[i%9]
            w = boxes['target_sz'][0]#.astype(np.float)
            h = boxes['target_sz'][1]
            xmin = boxes['target_pos'][0]
            ymin = boxes['target_pos'][1]
            target_pos, target_sz = np.array([xmin, ymin]), np.array([w, h])
            
            state = SiamRPN_init(impre, target_pos, target_sz, net, args.model)
            state = SiamRPN_track(state, imdetec)
#            print("box " +str(i) +" target_pos =" + str(state['target_pos']))
#            print("Box " +str(i) +" score =" + str(state['score']))
            if state['score'] > 0.7:
#                label = "ID : {:0>6d}".format(boxid)
#                current_axis.text(xmin, ymin+h, label, size='medium', color='white', bbox={'facecolor':color, 'alpha':1.0})
                
#save the result to json
                xmin_org = boxes['org_pos'][0]
                ymin_org = boxes['org_pos'][1]
                w_org = boxes['org_sz'][0]
                h_org = boxes['org_sz'][1]
                
            
                hypo = {
                'id':     classname,
                'x':      round(xmin_org,2),
                'y':      round(ymin_org,2),
                'width':  round(w_org,2),
                'height': round(h_org,2)                  
                }
                hypotheses.append(hypo)
                
            else:
                count = count + 1
                templist[i]['count'] = count
                
        frameitem = {
            # no num
            'class':      'frame',
            'timestamp':  int(img_name.split('.')[0]),
            'hypotheses': hypotheses
        }
        frames.append(frameitem)
    

        for i , box in enumerate(templist):
            if box['count'] > 2:
                templist.pop(i)
                


# save the result to json
                
fileitem = {
        'class': 'video',
        'frames': frames
    }

json.dump(fileitem, jf, indent=4)
jf.close
                
#save the result images to output                
#        out_img_path =os.path.join(out_path,img_name)
#        fig.savefig(out_img_path, bbox_inches='tight', dpi=120, pad_inches=0.0)
        

  
#
#print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
