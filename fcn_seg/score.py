from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import scipy.misc
import scipy.io as sio

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
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
    print '>>>', datetime.now(), 'test Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'test Iteration', iter, 'IU', iu
    print '>>>', datetime.now(), 'test Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'test Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist

def draw_layers(solver,save_dir,dataset, layers=[]):
    print '>>>', datetime.now(), 'Begin draw layers: ', layers
    solver.test_nets[0].share_with(solver.net)
    net = solver.test_nets[0]
    print '>>> save dir is', save_dir
    #for idx in dataset:
    net.forward()
    #loss = net.blobs['edge/loss'].data.flat[0]
    print 'loss && edge/loss is ',net.blobs['loss'].data.flat[0], net.blobs['edge/loss'].data.flat[0]
    # print 'edge/argmax blobs data is'
    # print net.blobs['edge/argmax'].data[0]
    # print 'edge/sobelX params is '
    # print net.params['edge/sobelX'][0].data   
    # print 'edge/sobelX blobs data is'
    # print net.blobs['edge/sobelX'].data[0]
    if save_dir:
        for layer in layers:
            #im = Image.fromarray((net.blobs[layer].data[0].argmax(0)*100).astype(np.uint8), mode='P')
            #im.save(os.path.join(save_dir, 'iter_{}'.format(solver.iter)+ '_layer_'+layer + '.png'))
            sio.savemat(os.path.join(save_dir, 'iter_{}'.format(solver.iter)+ '_layer_'+layer.replace("/","_") + '.mat'),{'data': net.blobs[layer].data[0]} )
            # scipy.misc.imsave(os.path.join(save_dir, 'iter_{}'.format(solver.iter)+ '_layer_'+layer.replace("/","_") + '.png'), (net.blobs[layer].data[0].argmax(0)))
            # if layer == 'edge/argmax':
            #     scipy.misc.imsave(os.path.join(save_dir, 'iter_{}'.format(solver.iter)+ '_layer_'+layer.replace("/","_") + '.png'), (net.blobs[layer].data[0][0,:,:]))
    