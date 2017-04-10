import sys
sys.path.insert(0,'../../caffe/python')
sys.path.append('../../')
import caffe
import surgery, score

import numpy as np
import os

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

caffe.set_device(0)
caffe.set_mode_gpu()
MODEL_FILE = '../../ilsvrc-nets/deploy.prototxt'
PRETRAINED = '../../ilsvrc-nets/bvlc_googlenet.caffemodel'
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# copy base weights for fine-tuning
#solver.net.copy_from(base_weights)
solver = caffe.SGDSolver('solver.prototxt')
solver.net.params['conv1/7x7_s2'][0].data[:,0:3:1,:,:] = net.params['conv1/7x7_s2'][0].data[:,:,:,:]
