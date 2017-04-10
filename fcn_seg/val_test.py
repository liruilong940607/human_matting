##  Eval XiJin's First model

import cv2
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,'./caffe/python')
import caffe
import os

import scipy.io as sio    
import matplotlib.pyplot as plt  
plt.rcParams['figure.figsize']=[15,15]
import time
import scipy.misc

INPUT_CHANNEL = 3
MEAN = [114.578, 115.294, 108.353]
DATASET = 'douyu_7100'
n_cl = 2

Mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
Std = 0.015625

# load net
DEPLOY = '/home/dalong/Workspace/human_matting/fcn_seg/xinjin_model/VGG16_1c_dep.prototxt'
CAFFEMODEL = '/home/dalong/Workspace/human_matting/fcn_seg/xinjin_model/vgg_co_1labels_minAug_3c_iter_11000.caffemodel'
net = caffe.Net(DEPLOY, CAFFEMODEL, caffe.TEST)

imglist = open('./data/'+DATASET+'/test.txt').readlines()
datanum = len(imglist)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n,n)

hist = 0

for imgname in imglist:
    imgname = imgname.strip()
    
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    img = cv2.imread('./data/'+DATASET+'/Images/'+imgname+'.png')
    img = cv2.resize(img,(192,320))
    img = np.float32(img)
    img -= Mean
    img *= Std
    img = img.transpose((2,0,1))
    
    # load label
    label = sio.loadmat('./data/'+DATASET+'/Labels/'+imgname+'.mat')['label']
    label = np.array(label, dtype=np.float32)
    
    if INPUT_CHANNEL==4:        
        # load 4-th channel
        cimg = cv2.imread('./data/'+DATASET+'/pred_xijin/{}.png_pred.png'.format(imgname),cv2.IMREAD_GRAYSCALE)
        if cimg is None:
            print 'none cimg!!!', imgname
            continue
        cm = np.ndarray((1,600,376),dtype=np.float32)
        cimg = np.float32(cimg)
        cimg = (cimg-128)*Std
        cm[0,:,:] = cimg
        
        
    net.blobs['data'].reshape(1,*img.shape)
    net.blobs['data'].data[...] = img

    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['mask/prob'].data[0]
    out = out.transpose().reshape((32,18,2))[:,:,1]
    out = cv2.resize(out*255,(368,640))
    out[np.where(out<255*0.3)] = 0
    out[np.where(out>=255*0.3)] = 1
    out = np.array(out, dtype=np.int64)
    
    this_hist =  fast_hist(label.flatten(),out.flatten(),n_cl)
    this_hist = np.array(this_hist, dtype=np.float32)
    this_iu = np.diag(this_hist) / (this_hist.sum(1) + this_hist.sum(0) - np.diag(this_hist))
    print imgname, this_iu, np.mean(this_iu) # first is neg iou, second is pos iou 
    hist += this_hist
    
hist = np.array(hist, dtype=np.float32)
iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
print '================================= result ================================='
print 'INPUT_CHANNEL: ',INPUT_CHANNEL
print 'MEAN', MEAN
print 'DEPLOY: ',DEPLOY
print 'CAFFEMODEL: ',CAFFEMODEL
print 'TEST DATASET: ',DATASET,
print 'TEST DATASET NUM: ',datanum
print 'IOU: ', iu, np.mean(iu)    




