import numpy as np
from PIL import Image
import sys
sys.path.insert(0,'./caffe/python')
import caffe
import os

import scipy.io as sio    
import matplotlib.pyplot as plt    
import numpy as np  
from PIL import Image
import time
import scipy.misc

# load net
net = caffe.Net('./experiment/exper_douyu2700_4input_edge/deploy.prototxt', './experiment/exper_douyu2700_4input_edge/snapshot/douyu_2700_train_iter_50000.caffemodel', caffe.TEST)

rootdir = './data/douyu_2700/Images/'

imglist = os.listdir(rootdir)
for img in imglist:
    if not img[-3:]=='png':
        continue
    print img
    start = time.clock()
    imshape = (368, 640, 4)
    in_batch = np.ndarray(shape=(1, imshape[2], imshape[1], imshape[0]), dtype=np.float32)

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(rootdir+img)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((114.578, 115.294, 108.353))
    in_ = in_.transpose((2,0,1))
    in_batch[0,0:3,:,:] = in_.copy()
    im = Image.open('./data/douyu_2700/pred_xijin/{}.png_pred.png'.format(img[:-4]))
    in_ = np.array(im, dtype=np.float32)
    in_ -= 128
    in_batch[0,3,:,:] = in_.copy()
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(*in_batch.shape)
    net.blobs['data'].data[...] = in_batch
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['edge/score'].data[0].argmax(axis=0)
    #out = net.blobs['score'].data[0]
    #print out.shape
    #score = net.blobs['score'].data[0]
    #_max = np.amax(score[1,:,:])
    #_min = np.amin(score[1,:,:])
    #hitmap = (score[1,:,:]-_min)/(_max-_min)*255
    #img_hitmap = Image.fromarray(hitmap)
    #if img_hitmap.mode != 'RGB':
    #    img_hitmap = img_hitmap.convert('RGB')
    # img_hitmap.save(rootdir+'/hitmap/'+img[:-4]+'.png')
    # plt.imshow(hitmap,cmap='gray') 
    # plt.axis('off')
    # plt.savefig()
    # sio.savemat(rootdir+'/fcn_score/'+img[:-4]+'.mat', {'score': net.blobs['score'].data[0]})   

    
    scipy.misc.imsave('./xinjin_model/douyu_2700/Images/'+img+'_mask3_edge.png', out)
