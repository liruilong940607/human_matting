#python solve.py 2>&1 | tee train.log
#../../caffe/build/tools/caffe time -model=./deploy.prototxt -gpu=1

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


# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
weights = '../../ilsvrc-nets/bvlc_googlenet.caffemodel'
# solver.net.copy_from(weights)

MODEL_FILE = '../../ilsvrc-nets/deploy.prototxt'
PRETRAINED = '../../ilsvrc-nets/bvlc_googlenet.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_layers=['upscore2','upsample-fused-16', 'upsample']
surgery.interp(solver.net, interp_layers)


# copy base weights for fine-tuning
#solver.net.copy_from(base_weights)
solver.net.params['conv1/7x7_s2'][0].data[:,0:3:1,:,:] = net.params['conv1/7x7_s2'][0].data[:,:,:,:]

layerkeys = ['conv2/3x3_reduce', 
'conv2/3x3', 
'inception_3a/1x1',
'inception_3a/3x3_reduce',
'inception_3a/3x3',
'inception_3a/5x5_reduce',
'inception_3a/5x5',
'inception_3a/pool_proj',
'inception_3b/1x1',
'inception_3b/3x3_reduce',
'inception_3b/3x3',
'inception_3b/5x5_reduce', 
'inception_3b/5x5',
'inception_3b/pool_proj',
'inception_4a/1x1',
'inception_4a/3x3_reduce',
'inception_4a/3x3',
'inception_4a/5x5_reduce',
'inception_4a/5x5',
'inception_4a/pool_proj',
'inception_4b/1x1',
'inception_4b/3x3_reduce',
'inception_4b/3x3',
'inception_4b/5x5_reduce',
'inception_4b/5x5',
'inception_4b/pool_proj',
'inception_4c/1x1',
'inception_4c/3x3_reduce',
'inception_4c/3x3',
'inception_4c/5x5_reduce',
'inception_4c/5x5',
'inception_4c/pool_proj',
'inception_4d/1x1',
'inception_4d/3x3_reduce',
'inception_4d/3x3',
'inception_4d/5x5_reduce',
'inception_4d/5x5',
'inception_4d/pool_proj',
'inception_4e/1x1',
'inception_4e/3x3_reduce',
'inception_4e/3x3',
'inception_4e/5x5_reduce',
'inception_4e/5x5',
'inception_4e/pool_proj',
'inception_5a/1x1',
'inception_5a/3x3_reduce',
'inception_5a/3x3',
'inception_5a/5x5_reduce',
'inception_5a/5x5',
'inception_5a/pool_proj',
'inception_5b/1x1',
'inception_5b/3x3_reduce',
'inception_5b/3x3',
'inception_5b/5x5_reduce',
'inception_5b/5x5',
'inception_5b/pool_proj']
for key in layerkeys:
	solver.net.params[key][0].data[...] = net.params[key][0].data[...]


# scoring
test = np.loadtxt('../../data/douyu_2700/test.txt', dtype=str)

# solver.restore('./snapshot/douyu2700_train_without_bn_iter_130000.solverstate');

for _ in range(2000):
    solver.step(500)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    print '================== Accuracy in test dataset ======================'
    score.seg_tests(solver, False, test, layer='score', gt='sem')
    print '=================================================================='
