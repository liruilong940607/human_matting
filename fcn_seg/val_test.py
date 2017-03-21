import sys,os
sys.path.append('/home/lucheng/Workspace_lrl/FCN/caffe/python')
sys.path.append('/home/lucheng/Workspace_lrl/FCN/fcn.berkeleyvision.org')
import caffe

import numpy as np
from PIL import Image
import cv2
import glob
import scipy.misc

model_def_file  = '../caffe_seg.prototxt';
model_file      = '../snapshot/flickr1800_train_iter_50000.caffemodel';

# load net
net = caffe.Net(model_def_file, model_file, caffe.TEST)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe

imdir = '/home/lucheng/Workspace_lrl/2017.3.1koutu/douyuTV_rand_select/'
imgs = glob.glob(imdir+'*.png')
imlist = []
for img in imgs:
    n = os.path.splitext(img)[0]
    if not (n.endswith('_t') or n.endswith('_p') or n.endswith('_mask')):
        imlist.append(img)

for imname in imlist:
    pf = os.path.splitext(imname)[0]
    im = Image.open(imname)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((114.578, 115.294, 108.353))
    in_ = in_.transpose((2,0,1))
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    scipy.misc.imsave(pf+'_mask.png',out)