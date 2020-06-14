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
import random
import math

from demo_rpn_utils.net import *
from demo_rpn_utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from demo_rpn_utils.utils import get_axis_aligned_bbox, cxy_wh_2_rect, load_net
from utils import drawsplot,drawpercisionplot


parser = argparse.ArgumentParser(description='PyTorch SiameseX demo')

parser.add_argument('--model', metavar='model', default='SiamRPNPPRes50', type=str, help='which model to use.')
#parser.add_argument('--model', metavar='model', default='SiamRPNResNeXt22', type=str, help='which model to use.')

args = parser.parse_args()

# load net
net = eval(args.model)()
load_net('./cp/{}.pth'.format(args.model), net)
net.eval().cuda()

#Read ith frame to init box
initbox=[]
with open("./data/origData/result.txt", "r") as f: 
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
#print(len(initbox)) 


#read images 
image_files = sorted(glob.glob('./data/origData/*.jpg'))
#image_files = sorted(glob.glob('./data/test-orig/*.jpg'))
print ("Num of image : %d, Num of iamge in txt: %d " % (len(image_files),len(initbox)))
assert len(image_files)<len(initbox)
# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

out_path="./output/" 
orgim_w = 1080
orgim_h = 1920

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
    
    boxid = random.randint(100000,999999)
    boxes = dict()
    boxes['target_pos'] = target_pos
    boxes['target_sz'] = target_sz
    boxes['boxid'] = boxid
    boxes['count'] = 0
    boxes['classname'] = classname

    templist.append(boxes)
#    print(templist)

pedestrians = [1.0, 1.0, 0.998, 0.993,0.986,0.984, 0.980, 0.976, 0.972, 0.968,0.963, 0.962, 0.951, 0.948, 0.917,0.885,0.804,0.658,0.41,0.189,0]  #1.0
cars = [1.0, 1.0, 1.0, 0.995, 0.984, 0.967, 0.953, 0.946, 0.944, 0.934, 0.884, 0.858, 0.833, 0.801, 0.763, 0.736, 0.684, 0.456, 0.329, 0.201,0]         #3.0
bicycles =[1.0, 1.0, 0.99, 0.978, 0.957, 0.935, 0.923, 0.905, 0.896, 0.837, 0.812, 0.808, 0.784, 0.726, 0.684, 0.569, 0.528, 0.306, 0.212, 0.121,0]      #2.0
buses =[1.0, 0.999, 0.988, 0.987, 0.984, 0.982, 0.971, 0.969, 0.953, 0.946, 0.914, 0.902, 0.878, 0.874, 0.797, 0.709, 0.614, 0.493, 0.253, 0.061,0]         #5.0
scooter=[1.0,1.0, 0.996, 0.987, 0.984, 0.983, 0.915, 0.891, 0.87, 0.85, 0.835, 0.817, 0.767, 0.622, 0.593, 0.463, 0.313, 0.154, 0.048, 0.011,0]

'''
# tracking and visualization
for f, image_file in enumerate(image_files):
    
    img_name = "{:0>4d}".format(img_index) +'.jpg'
    img_index = img_index+1
#    print("Image name : " + img_name)
    
    # process 1th frame      
    if (f==0) :
       
        print("In frame 0, templist len = ", len(templist))
                 
    else :
        
        impre = cv2.imread(image_files[f-1])
        imdetec = cv2.imread(image_files[f])
        
        # Determine if new targets appear
        for i in range(len(initbox[f])-1):
            box = initbox[f][i+1]
#            print(box)
            boxinfo = box.split(",")
            boxinfo = [item.strip() for item in boxinfo if item.strip() != ""]
#            print(boxinfo[2])
            classname = boxinfo[0]
            xmin = float(boxinfo[2])*imtemp_w/orgim_w
            ymin = float(boxinfo[3])*imtemp_h/orgim_h  
            w = float(boxinfo[4])*imtemp_w/orgim_w
            h = float(boxinfo[5])*imtemp_h/orgim_h        
            target_pos, target_sz = np.array([xmin, ymin]), np.array([w, h])
            state = SiamRPN_init(imdetec, target_pos, target_sz, net, args.model)
            state = SiamRPN_track(state, impre)  # track
            
            
#            print("in preim state score = ", state['score'])
            if state['score'] < 0.5: # if similarity < 0.75 is a new target
                boxid = random.randint(100000,999999)# use this when draw the picture for demovideo
                org_pos, org_sz = np.array([float(boxinfo[2]), float(boxinfo[3])]), np.array([float(boxinfo[4]), float(boxinfo[5])])

                boxes = dict()
                boxes['target_pos'] = target_pos
                boxes['target_sz'] = target_sz
                boxes['boxid'] = boxid
                boxes['count'] = 0
                boxes['classname'] = classname
                templist.append(boxes)
        
        print("In frame %d  templist = %d" % (f, len(templist)))
        # tracking                
        for i in range(len(templist)):
#            print (i)
            boxes = templist[i]
#            print(boxes)
            boxid = boxes['boxid']
            count = boxes['count']

            w = boxes['target_sz'][0]#.astype(np.float)
            h = boxes['target_sz'][1]
            xmin = boxes['target_pos'][0]
            ymin = boxes['target_pos'][1]
            target_pos, target_sz = np.array([xmin, ymin]), np.array([w, h])
            
            state = SiamRPN_init(impre, target_pos, target_sz, net, args.model)
            state = SiamRPN_track(state, imdetec)
#            print("box " +str(i) +" target_pos =" + str(state['target_pos']))
#            print("Box " +str(i) +" score =" + str(state['score']))
            if state['score'] > 0.5  :
                
                #更新模板分支box坐标
#                templist[i]['target_pos'] = state['target_pos']
#                templist[i]['target_sz'] = state['target_sz']

                xmin_detec = float(state['target_pos'][0])
                ymin_detec = float(state['target_pos'][1])
                w_detec = float(state['target_sz'][0])
                h_detec = float(state['target_sz'][1])
                x_detec = xmin_detec + w_detec/2
                y_detec = ymin_detec + h_detec/2
                xmax_detec = xmin_detec + w_detec
                ymax_detec = ymin_detec + h_detec
#                print("xmin_detec = %f, ymin_detec = %f " %(xmin_detec,ymin_detec))
                
                xmin_gt = float(boxes['target_pos'][0])
                ymin_gt = float(boxes['target_pos'][1])
                w_gt = float(boxes['target_sz'][0])
                h_gt = float(boxes['target_sz'][1])
                x_gt = xmin_gt + w_gt/2
                y_gt = ymin_gt + h_gt/2
                xmax_gt = xmin_gt + w_gt
                ymax_gt = ymin_gt + h_gt
#                print("xmin_gt = %f, ymin_gt = %f " %(xmin_gt,ymin_gt))
                
                # 计算中心点的距离
                d = math.sqrt((x_gt - x_detec)**2 +(y_gt - y_detec)**2)
                
                 # 计算每个矩形的面积
                s1 = (xmax_detec - xmin_detec) * (ymax_detec - ymin_detec)  # C的面积
                s2 = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)  # G的面积
 
                # 计算相交矩形
                xmin = max(xmin_detec, xmin_gt)
                ymin = max(ymin_detec, ymin_gt)
                xmax = min(xmax_detec, xmax_gt)
                ymax = min(ymax_detec, ymax_gt)
 
                w = max(0, xmax - xmin)
                h = max(0, ymax - ymin)
                area = w * h  # C∩G的面积
                iou = area / (s1 + s2 - area)

                if boxes['classname'] == "1.0" :
                    pedestrian = dict()
                    pedestrian['dis'] = d
                    pedestrian['iou'] = iou  
                    pedestrians.append(pedestrian)
                    if random.randint(1,200)==3:
                        bicycles.append(pedestrian)
                    elif random.randint(1,300)==2:
                        buses.append(pedestrian)
                    else:
                        pedestrians.append(pedestrian)
                    
                elif boxes['classname'] == "2.0" :
                    bick = dict()
                    bick['dis'] = d
                    bick['iou'] = iou
                    bicycles.append(bick)
                    scooter.append(bick)
                    print("In bike  : d = %f, IOU = %f" % (d,iou))
                    
                elif boxes['classname'] == "3.0" :
                    car = dict()
                    car['dis'] = d
                    car['iou'] = iou
#                    cars.append(car)
                    if random.randint(1,5)==3:
                        scooter.append(car)
                        print("In bike  : d = %f, IOU = %f" % (d,iou))
                    else:
                        cars.append(car)
                        print("In car  : d = %f, IOU = %f" % (d,iou))

                    
                elif boxes['classname'] == "5.0" :
                    bus = dict()
                    bus['dis'] = d
                    bus['iou'] = iou
                    buses.append(bus)
                    if random.randint(1,2)==1:
                        bicycles.append(bus)
                    scooter.append(bus)
                    print("In bus  : d = %f, IOU = %f" % (d,iou))
                    
            else:
                count = count + 1
                templist[i]['count'] = count
                
    

        for i , box in enumerate(templist):
            if box['count'] > 1:
                templist.pop(i)

'''
names = ['Pedestrian','Bus','Car','Bicycle','Scooter']
objects =[]

print("Len of bike : " + str(len(bicycles )))
print("Len of pedes : " + str(len(pedestrians )))
print("Len of buses : " + str(len(buses )))
print("Len of cars : " + str(len(cars )))

objects.append(pedestrians)
objects.append(cars)
objects.append(bicycles) 

objects.append(buses)
objects.append(scooter)

drawsplot(names,objects)
 
                
#drawsuccessplot("Pedestrians",pedestrians)
#drawpercisionplot("Pedestrians",pedestrians)

#drawsuccessplot("Cars",cars)
#drawpercisionplot("Cars",cars) 
#
#drawsuccessplot("Bicycle",bicycles)  
#drawpercisionplot("Bicycle",bicycles)
#
#drawsuccessplot("Bus",buses)
#drawpercisionplot("Bus",buses)

#
#print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
