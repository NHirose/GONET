#!/usr/bin/env python
# -*- coding: utf-8 -*-
# From Seigo Ito topic no doki

import rospy
import message_filters
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import sys
import cv2
import time
import numpy as np
import math
import numpy

#PIL関連
from image_converter import decode, encode
import ImageDraw
import Image as PILImage

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda, Variable
from chainer import function
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer.utils import type_check
from resnet152_hirose import ResNet152Layers

#import cnn_hirose_var
#import cnn_hirose2_var
#import cnn_hirose4_var

i = 0
j = 0
'''
model_file_gen = 'out_models_stereox2/dcgan_model_gen_95.h5'
model_file_dis = 'out_models_stereox2/dcgan_model_dis_95.h5'
model_file_invg = 'out_models_invg_stereo/dcgan_model_invg_17.h5'
model_file_fl = 'out_models_FL_stereo_resnet152_rec5_smooth1x/model_RDFresnet_RDF_870.h5'  #test 98.5
'''

model_file_gen = 'nn_model/dcgan_model_gen_132.h5'
model_file_dis = 'nn_model/dcgan_model_dis_132.h5'
model_file_invg = 'nn_model/dcgan_model_invg_50.h5'
#model_file_fl = 'nn_model/model_RDFresnet_RDF_L_1300.h5'  #test 98.5
model_file_fl = 'nn_model/model_RDFresnet_RDF_L_2500.h5'  #test 98.5

nz = 100
ratio = 0.1

#center of picture
#yoko
xc = 310
#tate
yc = 321

yoffset = 310 
xoffset = 310
xyoffset = 275
XYc = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]

# resize parameters
rsizex = 128
rsizey = 128

# zeros
outlist = np.zeros(15)

ofs=open("dataset_128y.txt","w")
#bag = '_bag1_'
#bag = '_test_'
#bag = '_test1_'
# callback function of ROS messages

'''
class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        _, h, w = image.shape
        imagex = (image - 128)/128

        return imagex, label, image
        #return image, label
'''

class Generator(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 8*8*512, initialW=initializer),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=initializer),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=initializer),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=initializer),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=initializer),
            bn0l = L.BatchNormalization(8*8*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        #print z.data.shape
        h = F.reshape(F.relu(self.bn0l(self.l0z(z))), (z.data.shape[0], 512, 8, 8))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = (self.dc4(h))
        return x

class invG(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(invG, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initializer),
            l4l = L.Linear(8*8*512, nz, initialW=initializer),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = F.relu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h))) 
        h = F.relu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return l

class ELU(function.Function):

    """Exponential Linear Unit."""
    # https://github.com/muupan/chainer-elu

    def __init__(self, alpha=1.0):
        self.alpha = numpy.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,

def elu(x, alpha=1.0):
    """Exponential Linear Unit function."""
    # https://github.com/muupan/chainer-elu
    return ELU(alpha=alpha)(x)

class Discriminator(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initializer),
            l4l = L.Linear(8*8*512, 2, initialW=initializer),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = elu(self.bn1(self.c1(h)))
        h = elu(self.bn2(self.c2(h))) 
        h = elu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return h

class FL(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(FL, self).__init__(
            l_img = L.Linear(3*128*128, 10, initialW=initializer),
            l_dis = L.Linear(512*8*8, 10, initialW=initializer),
            l_fdis = L.Linear(512*8*8, 10, initialW=initializer),
            #l_resnet = L.Linear(2048*7*7, 10, initialW=initializer),
            #l_sia = L.Linear(1024, 10, initialW=initializer),
            #l_FL = L.Linear(5, 1, initialW=initializer),
            l_LSTM = L.LSTM(30, 30),
            l_FL = L.Linear(30, 1, initialW=initializer),
            #bnimg = L.BatchNormalization(6*128*128),
            #bndis = L.BatchNormalization(512*8*8),
            #bnfdis = L.BatchNormalization(512*8*8),
            bnfl = L.BatchNormalization(2048*7*7),
            #bnfl = L.BatchNormalization(4096),
            #bnfl_sia = L.BatchNormalization(1024),
        )
    def reset_state(self):
        self.l_LSTM.reset_state()

    def set_state(self):
        self.l_LSTM.set_state()
        
    def __call__(self, img_error, dis_error, dis_output, test=False):
        h = F.reshape(F.sqrt(F.square(img_error)), (img_error.data.shape[0], 3*128*128))
        h = self.l_img(h)
        #h = self.l_img(self.bnimg(h))
        g = F.reshape(F.sqrt(F.square(dis_error)), (dis_error.data.shape[0], 512*8*8))
        g = self.l_dis(g)
        #g = self.l_dis(self.bndis(g))
        f = F.reshape(dis_output, (dis_output.data.shape[0], 512*8*8))
        f = self.l_fdis(f)
        #f = self.l_fdis(self.bnfdis(f))
        #v = F.reshape(F.sqrt(F.square(dis_vgg)), (dis_vgg.data.shape[0], 4096))
        #v = F.reshape(F.sqrt(F.square(dis_vgg)), (dis_vgg.data.shape[0], 512*7*7))
        #v = F.reshape(F.sqrt(F.square(dis_resnet)), (dis_resnet.data.shape[0], 2048*7*7))
        #v = self.l_resnet(self.bnfl(v))
        #s = self.l_sia(self.bnfl_sia(feat_sia))
        con = F.concat((h,g,f), axis=1)
        #print con.data.shape
        #print con.data.shape
        ls = self.l_LSTM(con)
        #print ls.data.shape
        ghf = F.sigmoid(self.l_FL(ls))
        return ghf

def callback(msg_1):
    global i
    global j
    global outlist
    j = j + 1
    if j == 1:
        i = i+1
        xd = np.zeros((3, 3, 128, 128), dtype=np.float32)
        # resize and crop image for msg_1
        cv2_msg_img = bridge.imgmsg_to_cv2(msg_1)
        cv_imgc = bridge.cv2_to_imgmsg(cv2_msg_img, 'rgb8')
        pil_img = encode(cv_imgc)
        fg_img = PILImage.new('RGBA', pil_img.size, (0, 0, 0, 255))
        draw=ImageDraw.Draw(fg_img)
        draw.ellipse(XYc, fill = (0, 0, 0, 0)) # 中心を抜く
        pil_img.paste(fg_img, (0, 0), fg_img.split()[3])
        img_msg = decode(pil_img)
        cv2_imgd = bridge.imgmsg_to_cv2(img_msg, 'rgb8')

        cv_cutimg = cv2_imgd[yc-xyoffset:yc+xyoffset, xc-xyoffset:xc+xyoffset]
        cv_cutimg = cv2.transpose(cv_cutimg)
        cv_cutimg = cv2.flip(cv_cutimg,1)
        #
        cv_resize1 = cv2.resize(cv_cutimg,(rsizex, rsizey))
        cv_resizex = cv_resize1.transpose(2, 0, 1)
        in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
        in_img1 = (in_imgcc1 - 128)/128
        in_imgcc1u = np.reshape(np.array([cv_resizex], dtype=np.uint8), (3,128,128))
        #
        # resize and crop image for msg_2
        '''
        cv2_msg_img = bridge.imgmsg_to_cv2(msg_2)
        cv_imgc = bridge.cv2_to_imgmsg(cv2_msg_img, 'rgb8')
        pil_img = encode(cv_imgc)
        fg_img = PILImage.new('RGBA', pil_img.size, (0, 0, 0, 255))
        draw=ImageDraw.Draw(fg_img)
        draw.ellipse(XYc, fill = (0, 0, 0, 0)) # 中心を抜く
        pil_img.paste(fg_img, (0, 0), fg_img.split()[3])
        img_msg = decode(pil_img)
        cv2_imgd = bridge.imgmsg_to_cv2(img_msg, 'rgb8')

        cv_cutimg = cv2_imgd[yc-xyoffset:yc+xyoffset, xc-xyoffset:xc+xyoffset]
        cv_cutimg = cv2.transpose(cv_cutimg)
        cv_cutimg = cv2.flip(cv_cutimg,1)
        #
        cv_resize2 = cv2.resize(cv_cutimg,(rsizex, rsizey))
        cv_resizex = cv_resize2.transpose(2, 0, 1)
        in_imgcc2 = np.array([cv_resizex], dtype=np.float32)
        in_img2 = (in_imgcc2 - 128)/128
        in_imgcc2u = np.reshape(np.array([cv_resizex], dtype=np.uint8), (3,128,128))
        '''
        #print in_imgcc1u.shape, in_imgcc1u.astype
        #img_nn_nRxx = chainer.links.model.vision.resnet.prepare(in_imgcc1u)
        #img_nn_nLxx = chainer.links.model.vision.resnet.prepare(in_imgcc2u)
        #img_nn_nRx = np.reshape(img_nn_nRxx, (1,3,224,224))
        #img_nn_nLx = np.reshape(img_nn_nLxx, (1,3,224,224))
        #in_img = np.concatenate((in_img1, in_img2), axis=1)
        in_img = in_img1
        #
        in_imgg = cuda.to_gpu(in_img) #for 6ch GONOGO
        #img_nn_nRg = cuda.to_gpu(img_nn_nRx) #for Resnet
        #img_nn_nLg = cuda.to_gpu(img_nn_nLx) #for Resnet
        img_real = Variable(in_imgg)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            img_gen = gen(invg(img_real))
            dis_real = dis(img_real)
            dis_gen = dis(img_gen)
            #feature_Rd = resnet(img_nn_nRg)
            #feature_Ld = resnet(img_nn_nLg)
            #feature_R = feature_Rd['res5']
            #feature_L = feature_Ld['res5']
            output = fl(img_real-img_gen, dis_real-dis_gen, dis_real)

            #noise = invg(Variable(in_imgg))
            #out_img = gen(noise)
        '''
        out_imgc = cuda.to_cpu(out_img.data)     
        #
        xd = xp.zeros((1, 3, 128, 128), dtype=np.float32)
        xrd = xp.zeros((1, 3, 128, 128), dtype=np.float32)   
        #
        xd[0,:,:,:] = in_imgg
        xrd[0,:,:,:] = out_img.data
        with chainer.using_config('train', False), chainer.no_backprop_mode(): 
            dc = dis(xd)
            drc = dis(xrd)
            output = fl(xd-xrd, dc-drc, dc)
        '''
        '''
        for k in range(4):
            mean_aveg[k] = mean_aveg[k+1]
        mean_aveg[4] = output.data
        out_gonogo = (mean_aveg[0] + mean_aveg[1] + mean_aveg[2] + mean_aveg[3] + mean_aveg[4])/5.0
        out_gonogoc = cuda.to_cpu(out_gonogo)
        '''
        out_gonogoc = cuda.to_cpu(output.data)
        '''
        #output real and generated image
        #print out_img.data.shape
        imgb = np.fmin(255.0, np.fmax(0.0, out_imgc*128+128))
        #imgb = out_img.data*128+128
        #print imgb.shape
        imgc = np.reshape(imgb, (3, 128, 128))
        imgd = imgc.transpose(1, 2, 0)
        imge = imgd.astype(np.uint8)
        imgn = bridge.cv2_to_imgmsg(cv_resize)
        imgm = bridge.cv2_to_imgmsg(imge)
        image_pub_1.publish(imgm)
        image_pub_2.publish(imgn)
        outave = 0.0
        '''
        '''
        outdata = 0.0
        L_invg = 0.0
        cnn_out.publish(outdata)
        cnn_label.publish(L_invg)
        '''
        cnn_out2.publish(out_gonogoc)
        j = 0

def output_nn(cv_ac):
    noise = invg(Variable(cv_ac), test='True')
    out_img = gen(noise, test='True')
    return out_img

xp = cuda.cupy
cuda.get_device(0).use()

gen = Generator()
dis = Discriminator()
invg = invG()
fl = FL()
#resnet = ResNet152Layers()
serializers.load_hdf5(model_file_invg, invg)
serializers.load_hdf5(model_file_gen, gen)
serializers.load_hdf5(model_file_dis, dis)
serializers.load_hdf5(model_file_fl, fl)

gen.to_gpu()
dis.to_gpu()
invg.to_gpu()
#resnet.to_gpu()
fl.to_gpu()

mean_ave = xp.zeros((5), dtype=np.float32)
mean_aveg = cuda.to_gpu(mean_ave)

#bridge = CvBridge()
# main function
if __name__ == '__main__':

    #initialize node
    rospy.init_node('sync_topic', anonymous=True)
    #subscribe of topics
    msg1_sub = rospy.Subscriber('/cv_camera_node/image_raw', Image, callback)

    #publisher of topics
    #image_pub_1 = rospy.Publisher('gen_image',Image,queue_size=10)
    #image_pub_2 = rospy.Publisher('raw_image',Image,queue_size=10)
    #cnn_out = rospy.Publisher('cnn_out',Float32,queue_size=10)
    cnn_out2 = rospy.Publisher('cnn_out2',Float32,queue_size=10)
    #cnn_label = rospy.Publisher('cnn_label',Float32,queue_size=10)
    #image_pub_2 = rospy.Publisher('sync_odom',Odometry,queue_size=10)

    bridge = CvBridge()
    # waiting callback
    print 'waiting message .....'
    rospy.spin()
