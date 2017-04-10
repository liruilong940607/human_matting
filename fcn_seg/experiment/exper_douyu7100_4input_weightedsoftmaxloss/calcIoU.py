from __future__ import division
import sys
sys.path.insert(0,'../../caffe/python')
sys.path.append('../../')
import caffe
import surgery
import numpy as np
import os
from datetime import datetime
from PIL import Image
import scipy.misc
import time

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    count = 0
    for idx in dataset:
        count+=1
        print idx, '{}/{}'.format(count,len(dataset))
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            out = net.blobs[layer].data[0].argmax(0)
            scipy.misc.imsave(os.path.join(save_dir, idx + '.png'),out)
            #im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            #im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    # for 2 class:
    # hist = | right-right  right-wrong |
    #        | wrong-right  wrong-wrong |
    # 

    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'test Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'test Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'test Iteration', iter, 'accuracy', acc
    print '>>>', datetime.now(), 'test Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'test Iteration', iter, 'IU', iu
    print '>>>', datetime.now(), 'test Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'test Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist


#python solve.py 2>&1 | tee train.log
#../../caffe/build/tools/caffe time -model=./deploy.prototxt -gpu=1
# init
caffe.set_device(0)
caffe.set_mode_gpu()

# scoring
test = np.loadtxt('../../data/douyu_7100/test.txt', dtype=str)

solver = caffe.SGDSolver('solver.prototxt')
solver.restore('./snapshot/douyu_7100_train_iter_40000.solverstate');

# N.B. metrics on the semantic labels are off b.c. of missing classes;
# score manually from the histogram instead for proper evaluation
print '================== Accuracy in test dataset ======================'
seg_tests(solver, False, test, layer='score', gt='sem')
print '=================================================================='
