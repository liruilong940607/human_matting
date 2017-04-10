# -*- coding: utf-8 -*-
import sys,os
sys.path.insert(0,'../caffe/python')
import cv2
import numpy as np
import glob
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

md = 'VGG16_1c_dep.prototxt'
mw = 'vgg_co_1labels_minAug_3c_iter_11000.caffemodel'
Mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
Std = 0.15625

net = caffe.Net(md,mw,caffe.TEST)

def predict(iname,img):
    
    img = np.float32(img)
    
    img -= Mean
    img *= Std
    print img.shape
    img = img.transpose((2,0,1))
    
    net.blobs['data'].reshape(1,*img.shape)
    net.blobs['data'].data[...] = img
    net.forward()
#     print '========================================'
    
#     print net.blobs['conv1_1'].data.shape
#     print net.blobs['conv2_1'].data.shape
#     print net.blobs['conv3_1'].data.shape
#     print net.blobs['conv4_1'].data.shape
#     print net.blobs['conv5_1'].data.shape
#     print net.blobs['edge/cls'].data.shape
#     print net.blobs['edge/cls/reshape'].data.shape
#     print net.blobs['mask/cls'].data.shape
#     print net.blobs['mask/cls/reshape'].data.shape
    
#     print '========================================='
    
    out = net.blobs['mask/prob'].data[0]
    edge = net.blobs['edge/prob'].data[0]
    
    out = out.transpose().reshape((32,18,2))[:,:,1]
    edge = edge.transpose().reshape((16,9,2))[:,:,1]
    #out[np.where(out<0.7)] = 0
    out = cv2.resize(out*255,(368,640))
    #out[np.where(out<255*0.3)] = 0
    cv2.imwrite(iname+'_pred.png',out)

print 'start'
if os.path.isfile(sys.argv[1]):
    iname = sys.argv[1]
    img = cv2.imread(iname+'_r.png')
    predict(iname,img)
else:
    imgs = glob.glob(sys.argv[1]+'/*.png')
    imlist = []
    for im in imgs:
        n = os.path.splitext(im)[0]
        if not (n.endswith('_t') or n.endswith('_p') or n.endswith('_mask') or n.endswith('_e') or n.endswith('_ev') or n.endswith('_m') or n.endswith('_mv') or n.endswith('_r') or n.endswith('_pred') or n.endswith('_pedge')):
            imlist.append(im)
    for im in imlist:
        iname = im
        img = cv2.imread(iname)
        print 'processing',iname
        img = cv2.resize(img,(192,320))
        predict('../data/douyu_7100/pred_xijin/'+iname.split('/')[-1],img)
        #predict('./test6/pred_xijin/'+iname.split('/')[-1],img)

