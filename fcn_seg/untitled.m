clear
clc

addpath('./caffe/matlab');

Mean = [114.578, 115.294, 108.353];% bgr
Std = 0.015625;

model_def_file = '/home/dalong/Workspace/human_matting/fcn_seg/xinjin_model/VGG16_1c_dep.prototxt'
model_file = '/home/dalong/Workspace/human_matting/fcn_seg/xinjin_model/vgg_co_1labels_minAug_3c_iter_11000.caffemodel'

caffe.set_mode_gpu();
caffe.set_device(0);

caffe.reset_all()
net = caffe.Net(model_def_file, model_file, 'test');

for i = 1:100
i

img = imread('./data/douyu_2700/Images/douyuTV_1000017_1488519011_0.png');
imgshape = size(img);
img = img(:,:,[3,2,1]);
img = imresize(img,[320,192],'bilinear');

img = double(img);
img(:,:,1) = img(:,:,1) - Mean(1);
img(:,:,2) = img(:,:,2) - Mean(2);
img(:,:,3) = img(:,:,3) - Mean(3);
img = img * Std;
    
img = permute(img,[2 1 3]);

[h1,w1,c1] = size(img);
net.blobs('data').reshape([h1,w1,c1,1]);
net.blobs('data').set_data(img);
net.forward_prefilled();
res = net.blobs('mask/prob').get_data();

res = res(:,2);
res = reshape(res',[18,32]);
res = res';
out = imresize(res*255,[640,368],'bilinear');
out(out<255*0.3) = 0;
out(out>=255*0.3) = 1;

end