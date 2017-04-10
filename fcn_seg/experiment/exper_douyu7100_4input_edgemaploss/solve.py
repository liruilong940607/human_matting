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
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_layers=['upscore2','upsample-fused-16', 'upsample']
surgery.interp(solver.net, interp_layers)




# copy base weights for fine-tuning
#solver.net.copy_from(base_weights)
solver.net.params['conv1/7x7_s2'][0].data[:,0:3:1,:,:] = net.params['conv1/7x7_s2'][0].data[:,:,:,:]
solver.net.params['edge/sobelX'][0].data[0,0,:,:] = [[0,0,0],[0,0,0],[0,0,0]]
solver.net.params['edge/sobelX'][0].data[0,1,:,:] = [[1,0,-1],[2,0,-2],[1,0,-1]]
solver.net.params['edge/sobelY'][0].data[0,0,:,:] = [[0,0,0],[0,0,0],[0,0,0]]
solver.net.params['edge/sobelY'][0].data[0,1,:,:] = [[1,2,1],[0,0,0],[-1,-2,-1]]

#solver.net.params['edge/score'][0].data[:,0:3:1,:,:] = net.params['conv1/7x7_s2'][0].data[:,:,:,:]
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

weights = '../exper_douyu2700_googlenet_batchsize_4input/snapshot/douyu_2700_train_iter_40000.caffemodel'
solver.net.copy_from(weights)

# scoring
test = np.loadtxt('../../data/douyu_7100/test.txt', dtype=str)

# solver.restore('./snapshot/douyu_2700_train_iter_50000.solverstate');
lastparam = []
keys = ['score2','upscore2','score-inception_4e/output','upsample','edge/sobelY']
for _ in range(2000):
    score.draw_layers(solver,'./draw/',test, layers=['edge/eltwise','edge/bn','edgemap','score','sem','score-inception_3b/output','score-inception_4e/output','score4','bigscore','edge/bn/scale'])
    # for key in keys:
    #     lastparam.append(solver.net.params[key][0].data[...])
    solver.step(200)
    # for i in range(0,len(keys)):
    #     print keys[i],lastparam[i]==solver.net.params[key][0].data[...]
    
    
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    #print '================== Accuracy in test dataset ======================'
    #score.seg_tests(solver, False, test, layer='score', gt='sem')
    #print '=================================================================='
    
    