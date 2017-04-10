##########################################
## Predict a image sequence
##########################################
import sys
sys.path.insert(0,'../caffe/python')
import numpy as np
from PIL import Image

import caffe
import os

import scipy.io as sio    
import matplotlib.pyplot as plt    
import numpy as np  
from PIL import Image

# load net
net = caffe.Net('','../experiment/exper_douyu2700_googlenet_batchsize_4input/snapshot/douyu_2700_train_iter_40000.caffemodel', caffe.TEST)

rootdir = './test5/'

imglist = os.listdir(rootdir+'frames')
for img in imglist:
	if not img[-3:]=='png':
		continue
	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	im = Image.open(rootdir+'/frames/'+img)
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array((114.578, 115.294, 108.353))
	in_ = in_.transpose((2,0,1))

	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_
	# run net and take argmax for prediction
	net.forward()
	out = net.blobs['score'].data[0].argmax(axis=0)
	score = net.blobs['score'].data[0]
	_max = np.amax(score[1,:,:])
	_min = np.amin(score[1,:,:])
	hitmap = (score[1,:,:]-_min)/(_max-_min)*255
	img_hitmap = Image.fromarray(hitmap)
	if img_hitmap.mode != 'RGB':
	    img_hitmap = img_hitmap.convert('RGB')
	img_hitmap.save(rootdir+'/hitmap/'+img[:-4]+'.png')
	# plt.imshow(hitmap,cmap='gray') 
	# plt.axis('off')
	# plt.savefig()
	sio.savemat(rootdir+'/fcn_score/'+img[:-4]+'.mat', {'score': net.blobs['score'].data[0]})   

	import scipy.misc
	scipy.misc.imsave(rootdir+'/predict/'+img[:-4]+'.png', out)
