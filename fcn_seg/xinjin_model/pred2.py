# -*- coding: utf-8 -*-
import sys,os
sys.path.insert(0,'../caffe/python')
import cv2
import numpy as np
import glob
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

md = 'googlenet_seg_fine_full.prototxt'
mw = 'googlenet_fine_full_5c_1l_weight_iter_28000.caffemodel'
Mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
Std = 0.015625

net = caffe.Net(md,mw,caffe.TEST)

def predict(iname,img, cimg):

    if img.shape[0] != 600 or img.shape[1]!=376:
        img = cv2.resize(img,(376,600))
        cimg = cv2.resize(cimg,(376,600))
    img = np.float32(img)
    
    img -= Mean
    img *= Std
    img = img.transpose((2,0,1))

    cm = np.ndarray((1,600,376),dtype=np.float32)
    cimg = np.float32(cimg)
    cimg = (cimg-128)*Std
    cm[0,:,:] = cimg

    net.blobs['data'].reshape(1,*img.shape)
    net.blobs['data'].data[...] = img
    net.blobs['comask'].reshape(1,*cm.shape)
    net.blobs['comask'].data[...] = cm

    net.forward()
    
    out = net.blobs['mask/prob'].data[0]

    out = out.transpose((1,2,0))[:,:,1]
    out[np.where(out<0.3)] = 0
    cv2.imwrite(iname+'_finePred.png',out*255)
if os.path.isfile(sys.argv[1]):
    iname = sys.argv[1]
    img = cv2.imread(iname)
    cimg = cv2.imread(iname + '_pred.png',cv2.IMREAD_GRAYSCALE)
    predict(iname,img,cimg)
else:
    imgs = glob.glob(sys.argv[1]+'/*.png')
    imlist = []
    for im in imgs:
        n = os.path.splitext(im)[0]
        if not (n.endswith('_t') or n.endswith('_p') or n.endswith('_mask') or n.endswith('_e') or n.endswith('_ev') or n.endswith('_m') or n.endswith('_mv') or n.endswith('_r') or n.endswith('_pred') or n.endswith('_pedge') or n.endswith('_f') or n.endswith('_fv') or n.endswith('_finePred') or n.endswith('_vdiff') or n.endswith('_vgrad') or n.endswith('_diff') or n.endswith('_grad') or n.endswith('_rf') or n.endswith('_vmask')):
            imlist.append(im)
    for im in imlist:
        iname = im
        img = cv2.imread(iname)
        cimg = cv2.imread(iname + '_pred.png',cv2.IMREAD_GRAYSCALE)
        if cimg is None:
            continue
        print 'processing',iname
        predict(iname,img, cimg)


