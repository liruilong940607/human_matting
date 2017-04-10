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
    
    loss = 0
    for i in range(0,10):
        idx = dataset[i]
        net.forward()
        if save_dir:
            for layer in layers:
                sio.savemat(os.path.join(save_dir, idx+'_iter_{}'.format(solver.iter)+ '_layer_'+layer.replace("/","_") + '.mat'),{'data': net.blobs[layer].data[0]} )
        loss += net.blobs['loss'].data.flat[0]
    print '>>>', datetime.now(), 'test Iteration', solver.iter, 'loss', loss/len(dataset)
            #im = Image.fromarray((net.blobs[layer].data[0].argmax(0)*100).astype(np.uint8), mode='P')
            #im.save(os.path.join(save_dir, 'iter_{}'.format(solver.iter)+ '_layer_'+layer + '.png'))
            #sio.savemat(os.path.join(save_dir, 'iter_{}'.format(solver.iter)+ '_layer_'+layer.replace("/","_") + '.mat'),{'data': net.blobs[layer].data[0]} )

def draw_layers_one_img(solver, save_dir, layers=[]):
    print '>>>', datetime.now(), 'Begin draw layers: ', layers
    solver.test_nets[0].share_with(solver.net)
    net = solver.test_nets[0]
    print '>>> save dir is', save_dir
    
    INPUT_CHANNEL = 4
    DATASET = 'douyu_7100'
    MEAN = [114.578, 115.294, 108.353]
    img = '660209_1489763036_0'
    
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open('../../data/'+DATASET+'/Images/'+img+'.png')
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array(MEAN)
    in_ = in_.transpose((2,0,1))
    
    # load label
    label = sio.loadmat('../../data/'+DATASET+'/Labels/'+img+'.mat')['label']
    label[label!=0]==1
    
    if INPUT_CHANNEL==4:
        imshape = (368, 640, 4)
        in_batch = np.ndarray(shape=(1, imshape[2], imshape[1], imshape[0]), dtype=np.float32)
        in_batch[0,0:3,:,:] = in_.copy()
        
        # load 4-th channel
        im_pred = Image.open('../../data/'+DATASET+'/pred_xijin/{}.png_pred.png'.format(img))
        in_ = np.array(im_pred, dtype=np.float32)
        in_ -= 128
        in_batch[0,3,:,:] = in_.copy()
        
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(*in_batch.shape)
    net.blobs['data'].data[...] = in_batch 
    net.forward()
    if save_dir:
        for layer in layers:
            sio.savemat(os.path.join(save_dir, img+'_iter_{}'.format(solver.iter)+ '_layer_'+layer.replace("/","_") + '.mat'),{'data': net.blobs[layer].data[0]} )
    loss = net.blobs['loss'].data.flat[0]
    print '>>>', datetime.now(), 'test Iteration', solver.iter, 'test img', img, 'loss', loss

            