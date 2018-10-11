#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import time
import numpy as np
import math
import numpy

#ROS
import rospy
import message_filters
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

#PIL
from image_converter import decode, encode
import ImageDraw
import Image as PILImage

#chainer
import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda, Variable
from chainer import function
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer.utils import type_check

i = 0
j = 0

model_file_gen = 'nn_model/featlayer_gen_single.h5'
model_file_dis = 'nn_model/featlayer_dis_single.h5'
model_file_invg = 'nn_model/featlayer_invg_single.h5'
model_file_fl = 'nn_model/classlayer_t.h5'

nz = 100

#center of picture
xc = 310
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
        h = F.relu(self.c0(x))
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
        h = elu(self.c0(x))
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
            l_LSTM = L.LSTM(30, 30),
            l_FL = L.Linear(30, 1, initialW=initializer),
            bnfl = L.BatchNormalization(2048*7*7),
        )
    def reset_state(self):
        self.l_LSTM.reset_state()

    def set_state(self):
        self.l_LSTM.set_state()
        
    def __call__(self, img_error, dis_error, dis_output, test=False):
        h = F.reshape(F.absolute(img_error), (img_error.data.shape[0], 3*128*128))
        h = self.l_img(h)
        g = F.reshape(F.absolute(dis_error), (dis_error.data.shape[0], 512*8*8))
        g = self.l_dis(g)
        f = F.reshape(dis_output, (dis_output.data.shape[0], 512*8*8))
        f = self.l_fdis(f)
        con = F.concat((h,g,f), axis=1)
        ls = self.l_LSTM(con)
        ghf = F.sigmoid(self.l_FL(ls))
        return ghf

def callback(msg_1):
    global i
    global j
    global outlist
    j = j + 1
    if j == 1:
        i = i+1
        # resize and crop image
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
        #
        #
        in_imgg = cuda.to_gpu(in_img1) # to gpu
        img_real = Variable(in_imgg)
        #
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            img_gen = gen(invg(img_real))
            dis_real = dis(img_real)
            dis_gen = dis(img_gen)
            output = fl(img_real-img_gen, dis_real-dis_gen, dis_real) #traversable probablity

        out_gonogoc = cuda.to_cpu(output.data) # to cpu

        gonet_out.publish(out_gonogoc) #publish
        j = 0

xp = cuda.cupy
cuda.get_device(0).use()

#definition of models
gen = Generator()
dis = Discriminator()
invg = invG()
fl = FL()

#load parameter of models
serializers.load_hdf5(model_file_invg, invg)
serializers.load_hdf5(model_file_gen, gen)
serializers.load_hdf5(model_file_dis, dis)
serializers.load_hdf5(model_file_fl, fl)

# sending models to gpu
gen.to_gpu()
dis.to_gpu()
invg.to_gpu()
fl.to_gpu()

if __name__ == '__main__':

    #initialize node
    rospy.init_node('sync_topic', anonymous=True)
    #subscribe of topics
    msg1_sub = rospy.Subscriber('/cv_camera_node/image_raw', Image, callback)

    #publisher of topics
    gonet_out = rospy.Publisher('out_GONet',Float32,queue_size=10)

    bridge = CvBridge()
    print 'waiting message .....'
    rospy.spin()
