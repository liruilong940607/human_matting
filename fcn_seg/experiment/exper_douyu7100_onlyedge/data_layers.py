import caffe

import numpy as np
from PIL import Image
import scipy.io
import time
import random

import cv2,os
MAX_RATE = 0.75
class Douyu1000SegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from douyuTV
    one-at-a-time while reshaping the net to preserve dimensions.

    This data layer has three tops:

    1. the data, pre-processed
    2. the semantic labels 0-1 and void 255

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - data_dir: path to SIFT Flow dir
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for semantic segmentation of object and geometric classes.

        example: params = dict(data_dir="/path/to/siftflow", split="val")
        """
        # config
        # douyu7100   b,g,r:  130.872875286 132.061371593 148.766416982
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.split = params['split']
        self.mean = np.array((148.766, 132.061, 130.873), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params.get('batch_size',16)
        self.flip = params.get('flip', 1)
        self.crop = params.get('crop', 1)

        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.data_dir, self.split)
        self.dataset = open(split_f, 'r').read().splitlines()
        self.datalen = len(self.dataset)
        self.indices = [i for i in xrange(self.datalen)]
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        if self.random:
            random.seed(self.seed)
            print 'shuffling'
            random.shuffle(self.indices)

        self.idx = 0
    
    def time_me(fn):
        def _wrapper(*args, **kwargs):
            start = time.clock()
            fn(*args, **kwargs)
            print "%s cost %s second"%(fn.__name__, time.clock() - start)
        return _wrapper
    
    def reshape(self, bottom, top):
        # load image + label image pair
        if self.flip:
            fliper = [random.random() > 0.5 for _ in xrange(self.batch_size)]
        else:
            fliper = [False for _ in xrange(self.batch_size)]
        if self.crop:
            cropper = [((random.random() > 0.5),random.random(),random.random(),random.random()) for _ in xrange(self.batch_size)]
        else:
            cropper = [(False,0,0,0) for _ in xrange(self.batch_size)]
        self.data = self.load_image(fliper,cropper)
        self.alabel = self.load_label(fliper,cropper)
        #self.edgemap = self.load_edgemap(fliper,cropper)
        self.edgemap = self.load_edge(fliper,cropper)
        self.idx += self.batch_size
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.alabel.shape)
        top[2].reshape(*self.edgemap.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.alabel
        top[2].data[...] = self.edgemap
        # pick next input
        if self.idx + self.batch_size -1 > self.datalen-1:
            if self.random:
                print 'shuffling'
                random.shuffle(self.indices)
            self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass
    
    def load_image(self,flip,crop):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        imshape = (368, 640, 4)
        in_batch = np.ndarray(shape=(self.batch_size, imshape[2], imshape[1], imshape[0]), dtype=np.float32)
        for i in xrange(self.batch_size):   
            im = Image.open('{}/Images/{}.png'.format(self.data_dir, self.dataset[self.indices[self.idx+i]]))
            if flip[i]:
                im = np.fliplr(im)
                im = Image.fromarray(np.uint8(im))
            if crop[i][0]:
                r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                ns = (im.size[0]*r,im.size[1]*r)
                hoff = int((im.size[1]-ns[1])*crop[i][2])
                woff = int((im.size[0]-ns[0])*crop[i][3])
                roi = im.crop([woff,hoff,int(woff+ns[0]),int(hoff+ns[1])])
                im = roi.resize(im.size)
            in_ = np.array(im, dtype=np.float32)
            in_ = in_[:,:,0:3]
            in_ = in_[:,:,::-1]
            in_ -= self.mean
            in_ = in_.transpose((2,0,1))
            in_batch[i,0:3,:,:] = in_.copy()
            ## pred_xijin
            im = Image.open('{}/pred_xijin/{}.png_pred.png'.format(self.data_dir, self.dataset[self.indices[self.idx+i]]))
            if flip[i]:
                im = np.fliplr(im)
                im = Image.fromarray(np.uint8(im))
            if crop[i][0]:
                r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                ns = (im.size[0]*r,im.size[1]*r)
                hoff = int((im.size[1]-ns[1])*crop[i][2])
                woff = int((im.size[0]-ns[0])*crop[i][3])
                roi = im.crop([woff,hoff,int(woff+ns[0]),int(hoff+ns[1])])
                im = roi.resize(im.size)
            in_ = np.array(im, dtype=np.float32)
            in_ -= 128
            in_batch[i,3,:,:] = in_.copy()
        return in_batch
    
    def load_label(self,flip,crop):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #label_batch = []
        imshape = (368, 640, 1)
        label_batch = np.ndarray(shape=(self.batch_size, imshape[2], imshape[1], imshape[0]), dtype=np.float32)
        if True:
            for i in xrange(self.batch_size):
                label = scipy.io.loadmat('{}/Labels/{}.mat'.format(self.data_dir, self.dataset[self.indices[self.idx+i]]))['label']
                im = Image.fromarray(np.uint8(label))
                if flip[i]:
                    im = np.fliplr(im)
                    im = Image.fromarray(np.uint8(im))
                if crop[i][0]:
                    r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                    ns = (im.size[0]*r,im.size[1]*r)
                    hoff = int((im.size[1]-ns[1])*crop[i][2])
                    woff = int((im.size[0]-ns[0])*crop[i][3])
                    roi = im.crop([woff,hoff,int(woff+ns[0]),int(hoff+ns[1])])
                    im = roi.resize(im.size,Image.NEAREST)
                label = np.array(im)
                label[label>1] = 1
                label = label.astype(np.uint8)
                label = label[np.newaxis, ...]
                label_batch[i,:,:,:] = label.copy()
                #label_batch.append(label)
        else:
            raise Exception("Unknown label type: {}. Pick semantic.".format(label_type))
        #label_batch = np.asarray(label_batch)
        
        return label_batch
    
    def load_edgemap(self,flip,crop):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        imshape = (368, 640, 1)
        label_batch = np.ndarray(shape=(self.batch_size, imshape[2], imshape[1], imshape[0]), dtype=np.float32)
        if True:
            for i in xrange(self.batch_size):
                label = scipy.io.loadmat('{}/edgemap/{}.mat'.format(self.data_dir, self.dataset[self.indices[self.idx+i]]))['edgemap']
                img = label
                if flip[i]:
                    img = np.fliplr(img)
                if crop[i][0]:
                    r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                    ns = (img.shape[0]*r, img.shape[1]*r)
                    hoff = int((img.shape[0]-ns[0])*crop[i][2])
                    woff = int((img.shape[1]-ns[1])*crop[i][3])
                    roi = img[hoff:hoff+ns[0],woff:woff+ns[1]]
                    img = cv2.resize(roi,(img.shape[1],img.shape[0]))
                label = np.float32(img)
                ## TODO: consider other normalize function 
                max_ = np.max(label)
                min_ = np.min(label)
                if max_==min_:
                    label = np.zeros(label.shape)+1
                else:
                    label = (label-min_)/(max_-min_)
                label -= 0.8
                #label = 1- 1.0 / (1 + np.exp(-label));  
                label = label[np.newaxis, ...]
                label_batch[i,:,:,:] = label.copy()
        else:
            raise Exception("Unknown label type: {}. Pick semantic.".format(label_type))
        
        return label_batch
        
    def load_edge(self,flip,crop):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #label_batch = []
        imshape = (368, 640, 1)
        label_batch = np.ndarray(shape=(self.batch_size, imshape[2], imshape[1], imshape[0]), dtype=np.float32)
        if True:
            for i in xrange(self.batch_size):
                im = Image.open('{}/edge/{}.png_e.png'.format(self.data_dir, self.dataset[self.indices[self.idx+i]]))
                if flip[i]:
                    im = np.fliplr(im)
                    im = Image.fromarray(np.uint8(im))
                if crop[i][0]:
                    r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                    ns = (im.size[0]*r,im.size[1]*r)
                    hoff = int((im.size[1]-ns[1])*crop[i][2])
                    woff = int((im.size[0]-ns[0])*crop[i][3])
                    roi = im.crop([woff,hoff,int(woff+ns[0]),int(hoff+ns[1])])
                    im = roi.resize(im.size,Image.NEAREST)
                label = np.array(im)
                label[label>0] = 1
                label = label.astype(np.uint8)
                label = label[np.newaxis, ...]
                label_batch[i,:,:,:] = label.copy()
                #label_batch.append(label)
        else:
            raise Exception("Unknown label type: {}. Pick semantic.".format(label_type))
        #label_batch = np.asarray(label_batch)
        
        return label_batch