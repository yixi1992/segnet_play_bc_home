import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from PIL import Image
from sklearn.preprocessing import normalize
caffe_root = '/scratch/groups/lsdavis/yixi/software/caffe-segnet/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


if True:
	net = caffe.Net('/home-4/yixi@umd.edu/work/yixi/segnet/repcamvid/segnet_basic_train.prototxt','/home-4/yixi@umd.edu/work/yixi/segnet/repcamvid/snapshots/rtlr0.1_iter_11100.caffemodel',caffe.TEST)
	net.forward()
	np.save('norm0',np.array(net.blobs['norm'].data))
	np.save('conv10',np.array(net.params['conv1'][0].data))



if False:
	net = caffe.Net('ftrgb_inference.prototxt', 'snapshots_whole/ftrgbgglr1e-4fixed_iter_3000.caffemodel', caffe.TEST)
	net.forward()
	np.save('label',np.array(net.blobs['label'].data))
	np.save('conv_classifier',np.array(net.blobs['conv_classifier'].data))
	np.save('per_class_accuracy',np.array(net.blobs['per_class_accuracy'].data))

if False:
	net = caffe.Net('segnet_basic_train.prototxt','snapshots/gglr1e-5fixedadagrad_iter_1600.caffemodel', caffe.TEST)
	net.forward()
	np.save('conv1',np.array(net.blobs['conv1'].data))
	np.save('conv1_flow0', np.array(net.params['conv1_flow'][0].data))
	np.save('conv1_flow1', np.array(net.params['conv1_flow'][1].data))

if False:
	net5=caffe.Net('segnet_basic_train.prototxt', 'basic_camvid_surg.caffemodel', caffe.TEST)
	net5.forward()
	net=net5
	im = np.array(np.squeeze(net.blobs['data'].data), dtype=np.uint8)
	label = np.array(net.blobs['label'].data, dtype=np.uint8)
	norm = np.array(net.blobs['norm'].data)
	flow = np.array(net.blobs['flow'].data)
	np.save('flow3',flow)
	np.save('im3',im)
	np.save('label3',label)
	np.save('norm3',norm)
	np.save('normflow3',np.array(net.blobs['normflow'].data))

if False:
	net4=caffe.Net('segnet_basic_train1.prototxt', 'basic_camvid_surg.caffemodel', caffe.TEST)
	net4.forward()
	im = np.array(np.squeeze(net4.blobs['data'].data), dtype=np.uint8)
	label = np.array(net4.blobs['label'].data, dtype=np.uint8)
	norm = np.array(net4.blobs['norm'].data)
	np.save('im1',im)
	np.save('label1',label)
	np.save('norm1',norm)
	np.save('flow1',np.array(net4.blobs['flow'].data))
	np.save('normflow1',np.array(net4.blobs['normflow'].data))

print 'success!'
