# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys,os

import glob
import subprocess
imdir = sys.argv[1]
imgs = glob.glob(imdir+'*.png')
imlist = []
for im in imgs:
    n = os.path.splitext(im)[0]
    if not (n.endswith('_t') or n.endswith('_p') or n.endswith('_mask') or n.endswith('_e') or n.endswith('_ev') or n.endswith('_m') or n.endswith('_mv') or n.endswith('_r') or n.endswith('_pred') or n.endswith('_pedge')):
        imlist.append(im)
iouData = []
iouName = []
#SF = '_mask.png'
#SF = '_pred.png'
SF = '_finePred.png'
sia = [0,0]
sua = [0,0]
for im in imlist:
    pf = os.path.splitext(im)[0]
    if os.path.isfile(im+SF):
        mask = np.float32(cv2.imread(im+SF,0))
        mask = cv2.resize(mask,(368,640))
        mask[mask>0]=1
        gimg = cv2.imread(pf+'_p.png',-1)
        gimg = cv2.resize(gimg,(368,640))        
        gt = np.float32(gimg[:,:,3])
        if os.path.isfile(pf+'_t.png'):
            gimg = cv2.imread(pf+'_t.png',-1)
            gimg = cv2.resize(gimg,(368,640))        
            egt = np.float32(gimg[:,:,3])
            gt = gt+egt
        gt[gt>0] = 1
        try:
            interM = mask * gt
            unionM = mask + gt
        except:
            print 'fuck',im
            continue
        iA_plus = np.count_nonzero(interM)
        uA_plus = np.count_nonzero(unionM)
        try:
            interM = (1-mask) * (1-gt)
            unionM = (1-mask) + (1-gt)
        except:
            print 'fuck',im
            continue
        iA_minus = np.count_nonzero(interM)
        uA_minus = np.count_nonzero(unionM)
        
        IoU = np.float32(iA_plus)/uA_plus
        sia[0] += iA_plus
        sia[1] += iA_minus
        sua[0] += uA_plus
        sua[1] += uA_minus
        print im,IoU, sia
        iouData.append(IoU)
        iouName.append(im)
sia = np.array(sia, dtype=np.float32)
sua = np.array(sua, dtype=np.float32)
print sia[0]/sua[0], sia[1]/sua[1], np.mean([sia[0]/sua[0], sia[1]/sua[1]])
#print len(iouData),np.max(iouData),np.min(iouData),np.mean(iouData),np.median(iouData),\
#    np.percentile(iouData, 1),np.percentile(iouData, 10),np.percentile(iouData, 25),np.float32(sia)/sua
