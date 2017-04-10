import caffe

import numpy as np
from PIL import Image
import scipy.io

import random

import cv2,os
MAX_RATE = 0.75
class PSCDataLayer(caffe.Layer):
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

        - flickr1800_dir: path to SIFT Flow dir
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for semantic segmentation of object and geometric classes.

        example: params = dict(flickr1800_dir="/path/to/siftflow", split="val")
        """
        # config
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
        self.std = 0.15625
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params.get('batch_size', 1)
        self.list_name = params.get('list_name', 'list.txt')
        self.flip = params.get('flip', 0)
        self.crop = params.get('crop', 0)
        self.aux = params.get('aux', 1)
        
        # two tops: data, semantic
        if not self.aux:
            if len(top) != 3:
                raise Exception("Need to define 2 tops: data, mask label")
        else:
            if len(top) != 6:
                raise Exception("Need to define 6 tops: data, mask label, l11,l12,l21,l22.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        #read the list
        lines = open(self.data_dir + '/'+self.list_name,'r').readlines()
        self.dataset = []
        for line in lines:
            sp = line.strip().split()
            if len(sp) == 7:
                self.dataset.append((self.data_dir +'/' + sp[0],self.data_dir +'/' + sp[1],int(sp[2]),int(sp[3]),int(sp[4]),int(sp[5]),self.data_dir +'/' + sp[6]))
        self.datalen = len(self.dataset)
        print self.datalen,'samples read'
        self.indices = [i for i in xrange(self.datalen)]
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
        top[1].reshape(*self.alabel[0].shape)
        top[2].reshape(*self.alabel[5].shape)
        if self.aux:
            top[2].reshape(self.batch_size,1)
            top[3].reshape(self.batch_size,1)
            top[4].reshape(self.batch_size,1)
            top[5].reshape(self.batch_size,1)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.alabel[0]
        top[2].data[...] = self.alabel[5]
        if self.aux:
            top[2].data[...] = self.alabel[1]
            top[3].data[...] = self.alabel[2]
            top[4].data[...] = self.alabel[3]
            top[5].data[...] = self.alabel[4]
        # pick next input
        if self.idx + self.batch_size -1 > self.datalen-1:
            if self.random:
                print 'shuffling'
                random.shuffle(self.indices)
            self.idx = 0
        
        #self.data = self.load_image()
        #self.alabel = self.load_label()
        
        #self.idx += self.batch_size

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
        #print 'reading from',self.idx
        in_ = []
        for i in xrange(self.batch_size):
            img = cv2.imread(self.dataset[self.indices[self.idx+i]][0])
            if flip[i]:
                img = np.fliplr(img)
            if crop[i][0]:
                r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                ns = (img.shape[0]*r, img.shape[1]*r)
                hoff = int((img.shape[0]-ns[0])*crop[i][2])
                woff = int((img.shape[1]-ns[1])*crop[i][3])
                roi = img[hoff:hoff+ns[0],woff:woff+ns[1],:]
                img = cv2.resize(roi,(img.shape[1],img.shape[0]))
            img = np.float32(img)
            img -= self.mean
            img *= self.std
            img = img.transpose((2,0,1))
            in_.append(img)
        in_ = np.asarray(in_)
        return in_

    def load_label(self,flip,crop):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #label = np.resize(label,(600,800))
        al = [[] for _ in xrange(6)]
        for i in xrange(self.batch_size):
            img = cv2.imread(self.dataset[self.indices[self.idx+i]][1],cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dataset[self.indices[self.idx+i]][6],cv2.IMREAD_GRAYSCALE)
            if flip[i]:
                img = np.fliplr(img)
                img2 = np.fliplr(img2)
            if crop[i][0]:
                r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                ns = (img.shape[0]*r, img.shape[1]*r)
                hoff = int((img.shape[0]-ns[0])*crop[i][2])
                woff = int((img.shape[1]-ns[1])*crop[i][3])
                roi = img[hoff:hoff+ns[0],woff:woff+ns[1]]
                img = cv2.resize(roi,(img.shape[1],img.shape[0]),interpolation = cv2.INTER_NEAREST)
            if crop[i][0]:
                r = crop[i][1]*(1-MAX_RATE)+MAX_RATE
                ns = (img2.shape[0]*r, img2.shape[1]*r)
                hoff = int((img2.shape[0]-ns[0])*crop[i][2])
                woff = int((img2.shape[1]-ns[1])*crop[i][3])
                roi = img2[hoff:hoff+ns[0],woff:woff+ns[1]]
                img2 = cv2.resize(roi,(img2.shape[1],img2.shape[0]),interpolation = cv2.INTER_NEAREST)
            al[0].append(img.flatten())
            for j in xrange(4):
                al[j+1].append(self.dataset[self.indices[self.idx+i]][j+2])
            al[5].append(img2.flatten())
            if flip[i]:
                tmp = al[1][-1]
                al[1][-1] = al[2][-1]
                al[2][-1] = tmp
                tmp = al[3][-1]
                al[3][-1] = al[4][-1]
                al[4][-1] = tmp

        for i in xrange(6):
            al[i] = np.asarray(al[i],dtype=np.int32)
        for i in xrange(1,5):
            al[i]= al[i].reshape(self.batch_size,1)
        return al
