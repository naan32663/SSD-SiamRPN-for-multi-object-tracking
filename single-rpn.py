# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:02:01 2020

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

from demo_rpn_utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from demo_rpn_utils.utils import load_net

parser = argparse.ArgumentParser(description='PyTorch SiameseX demo')
parser.add_argument('--model', metavar='model', default='SiamRPNPPRes50', type=str, help='which model to use.')

#Get result from SSD
def readResult(path):
    initbox=[]
    with open(path, "r") as f: 
    # read the contents of file line by line: 
        for line in f: 
     # split current line by comma to get array of items from it 
         array_parts = line.split(";") 
     # filter elements, removing empty, and stripping spaces: 
         array_parts = [item.strip() for item in array_parts if item.strip() != ""] 
     # add line's items into result array: 
         initbox.append(array_parts)
    return initbox

#Get the bounding boxes in first frame
def getInitTemplist(initbox):
    templist = []
    for i in range(len(initbox[0])-1):
        box = initbox[0][i+1]

        boxinfo = box.split(",")
        boxinfo = [item.strip() for item in boxinfo if item.strip() != ""]
        #    print(boxinfo[2])
        classname = boxinfo[0]
        xmin = float(boxinfo[2])
        ymin = float(boxinfo[3])
        w = float(boxinfo[4])
        h = float(boxinfo[5])
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
    return templist


if __name__ == '__main__':
    
    args = parser.parse_args()

    # load net
    net = eval(args.model)()
    load_net('./cp/{}.pth'.format(args.model), net)
    net.eval().cuda()
    
    orgim1_w = 1080
    orgim1_h = 1920
    
    #read images 
    image_files1 = sorted(glob.glob('D:/120/SiameseX.PyTorch-master/data/test4/*.jpg'))


    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    out_path1="./output/" 

    classes = ['background', 'pedestrian','bicycle','car','motorcycle','bus','truck','scooter']
    
    initbox1 = readResult("D:/120/SiameseX.PyTorch-master/data/test4/result.txt")
    templist1=getInitTemplist(initbox1)
    

    # tracking and visualization the first video
    for f, image_file in enumerate(image_files1):
    
        img_name = str(image_file).split('\\')[1].split('.')[0]
#        print("Image name : " + img_name)
    
        # process 1th frame      
        if (f==0) :

            imdetec = cv2.imread(image_file)
#            print (imdetec.shape)
            imtemp_w = imdetec.shape[1]
            imtemp_h = imdetec.shape[0]
    
            plt.figure(figsize=(20,12))
            fig = plt.gcf()
            plt.imshow(imdetec)
            current_axis = plt.gca()
            plt.axis('off')
        
            print("In frame 0, templist1 len = ", len(templist1))
            for i in range(len(templist1)):
                boxes = templist1[i]
#            print(boxes)
                boxid = boxes['boxid']
                color = colors[i]
                w = float(boxes['target_sz'][0])*imtemp_w/orgim1_w#.astype(np.float)
                h = float(boxes['target_sz'][1])*imtemp_h/orgim1_h
                xmin = float(boxes['target_pos'][0])*imtemp_w/orgim1_w
                ymin = float(boxes['target_pos'][1])*imtemp_h/orgim1_h
                
                label = "ID: {:0>6d}".format(boxid)  
                current_axis.text(xmin, ymin+h, label, size='medium', color='white', bbox={'facecolor':color, 'alpha':1.0})  

            out_img_path =os.path.join(out_path1,img_name)
            fig.savefig(out_img_path, bbox_inches='tight', dpi=120, pad_inches=0.0)
        
        #process nth frames      
        else :

            impre = cv2.imread(image_files1[f-1])
            impre_w = impre.shape[1]
            impre_h = impre.shape[0]
            
            imdetec = cv2.imread(image_files1[f])
            imdetec_w = imdetec.shape[1]
            imdetec_h = imdetec.shape[0]
            
            detecObj = initbox1[f][1]
            
        # get the detec image
            plt.figure(figsize=(20,12))
            fig = plt.gcf()
            plt.imshow(imdetec)
            current_axis = plt.gca()
            plt.axis('off')
        
            # tracking                
            for i in range(len(templist1)):
#            print (i)
                boxes = templist1[i]
#            print(boxes)
                boxid = boxes['boxid']
                count = boxes['count']
                color = colors[i%9]
                w = float(boxes['target_sz'][0])*impre_w/orgim1_w#.astype(np.float)
                h = float(boxes['target_sz'][1])*impre_h/orgim1_h
                xmin = float(boxes['target_pos'][0])*impre_w/orgim1_w
                ymin = float(boxes['target_pos'][1])*impre_h/orgim1_h
                target_pos, target_sz = np.array([xmin, ymin]), np.array([w, h])
            
                state = SiamRPN_init(impre, target_pos, target_sz, net, args.model)
                
                boxinfo = detecObj.split(',')
#                print(boxinfo)
                boxinfo = [item.strip() for item in boxinfo if item.strip() != ""]
#                print(boxinfo[2])
                classname = boxinfo[0]
                xmin2 = float(boxinfo[2])*impre_w/orgim1_w
                ymin2 = float(boxinfo[3])*impre_h/orgim1_h
                w2 = float(boxinfo[4])*impre_w/orgim1_w
                h2 = float(boxinfo[5])*impre_h/orgim1_h 
                detec_pos, detec_sz = np.array([xmin2, ymin2]), np.array([w2, h2])  
        
                state = SiamRPN_track(state, imdetec,detec_pos, detec_sz)
#                state = SiamRPN_track(state, imdetec)
                
#            print("box " +str(i) +" target_pos =" + str(state['target_pos']))
                print("Box " +str(i) +" score =" + str(state['score']))
                if state['score'] > 0.5:
                    label = "ID : {:0>6d}".format(boxid)
                    current_axis.text(xmin2, ymin2 + h2, label, size='medium', color='white', bbox={'facecolor':color, 'alpha':1.0})
                
                else:
                    count = count + 1
                    templist1[i]['count'] = count    

                              
            out_img_path =os.path.join(out_path1,img_name)
            fig.savefig(out_img_path, bbox_inches='tight', dpi=120, pad_inches=0.0)