#python solve.py 2>&1 | tee train.log
#../../caffe/build/tools/caffe time -model=./deploy.prototxt -gpu=1

import sys
sys.path.insert(0,'../../caffe/python')
sys.path.append('../../')
import caffe
import surgery, score
import scipy.io as sio
import numpy as np
import os
import evaltrain

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass


# init
caffe.set_device(1)
caffe.set_mode_gpu()


#weights = '../../ilsvrc-nets/bvlc_googlenet.caffemodel'
# solver.net.copy_from(weights)

MODEL_FILE = '../../ilsvrc-nets/deploy.prototxt'
PRETRAINED = '../../ilsvrc-nets/bvlc_googlenet.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

solver = caffe.SGDSolver('solver.prototxt')
# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#interp_layers=['upscore2','upsample-fused-16', 'upsample']
surgery.interp(solver.net, interp_layers)


# copy base weights for fine-tuning
#solver.net.copy_from(base_weights)
solver.net.params['conv1/7x7_s2'][0].data[:,0:3:1,:,:] = net.params['conv1/7x7_s2'][0].data[:,:,:,:]

solver.net.params['score/slice_1/sobelx'][0].data[0,0,:,:] = [[1,0,1],[2,0,2],[1,0,1]]
solver.net.params['score/slice_1/sobely'][0].data[0,0,:,:] = [[1,2,1],[0,0,0],[1,2,1]]
solver.net.params['score/slice_2/sobelx'][0].data[0,0,:,:] = [[-1,0,1],[-2,0,2],[-1,0,1]]
solver.net.params['score/slice_2/sobely'][0].data[0,0,:,:] = [[-1,-2,-1],[0,0,0],[1,2,1]]

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
]
for key in layerkeys:
	solver.net.params[key][0].data[...] = net.params[key][0].data[...]


#weights = '../exper_douyu2700_googlenet_batchsize_4input/snapshot/douyu_2700_train_iter_40000.caffemodel'
#solver.net.copy_from(weights)

# scoring
test = np.loadtxt('../../data/douyu_7100/test.txt', dtype=str)

#solver.restore('./snapshot/douyu_7100_train_4input_iter_10000.solverstate');

for _ in range(2000):
    #evaltrain.draw_layers_one_img(solver,'./draw/', layers=['score-inception_3b/output','bigscore','score','sem','data'])
    #evaltrain.draw_layers_one_img(solver,'./draw/', layers=['score-inception_3b/output/edge','bigscore/edge','score/edge','edgemap'])
    #evaltrain.draw_layers_one_img(solver,'./draw/', layers=['score','sem','score/slice_1','score/slice_1/sobely','score/slice_1/sobelx','score/slice_1/sobely/abs','score/slice_1/sobelx/abs','score/slice_1/eltwise','score/sobel/concat'])
    solver.step(200)
    #print '================== Accuracy in test dataset ======================'
    score.seg_tests(solver, False, test, layer='score', gt='sem')
    #print '=================================================================='
    
    