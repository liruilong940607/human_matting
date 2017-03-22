import caffe

import numpy as np
from PIL import Image
import scipy.io

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
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.split = params['split']
        self.mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params.get('batch_size',16)
        self.flip = params.get('flip', 0)
        self.crop = params.get('crop', 0)

        # two tops: data, semantic
        if len(top) != 2:
            raise Exception("Need to define two tops: data, semantic label.")
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
        self.idx += self.batch_size
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.alabel.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.alabel
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
        in_batch = []
        for i in xrange(self.batch_size):
            im = Image.open('{}/Images/{}.png'.format(self.data_dir, self.dataset[self.indices[self.idx+i]]))
            #im.save('test_resize.png')
            if flip[i]:
                im = np.fliplr(im)
                im = Image.fromarray(np.uint8(im))
            if crop[i][0]:
                r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                #print self.dataset[self.indices[self.idx+i]], flip[i], im.size
                ns = (im.size[0]*r,im.size[1]*r)
                hoff = int((im.size[1]-ns[1])*crop[i][2])
                woff = int((im.size[0]-ns[0])*crop[i][3])
                roi = im.crop([woff,hoff,int(woff+ns[0]),int(hoff+ns[1])])
                im = roi.resize(im.size)
                #im.save('test_resize_%d_%d_%d_%d.png'%(woff,hoff,woff+ns[0],hoff+ns[1]))
            #if not (im.size[0]==368 and im.size[1]==640):   
            #im = im.resize((800,600))
            in_ = np.array(im, dtype=np.float32)
            in_ = in_[:,:,::-1]
            in_ -= self.mean
            in_ = in_.transpose((2,0,1))
            in_batch.append(in_)
        in_batch = np.asarray(in_batch)
        #print "image_shape: "
        #print in_batch.shape
        return in_batch

    def load_label(self,flip,crop):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label_batch = []
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
                label[label==2] = 1
                label = label.astype(np.uint8)
                label = label[np.newaxis, ...]
                label_batch.append(label)
        else:
            raise Exception("Unknown label type: {}. Pick semantic.".format(label_type))
        label_batch = np.asarray(label_batch)
        #print "label_shape: "
        #print label_batch.shape
        return label_batch