import pickle
import numpy as np
from PIL import Image
import os
from StringIO import StringIO
import math
import pylab


import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L

import numpy

xp = np

nz = 100
batchsize=100
n_epoch=10000
n_train=200000

#model_file_gen = 'out_models_Lx/dcgan_model_gen_60.h5'
#model_file_dis = 'out_models_Lx/dcgan_model_dis_60.h5'
#model_file_invg = 'out_models_invg_Ly/dcgan_model_invg_25.h5'
#model_file_fl = 'out_models_FL_L/model_best_RDF2.h5'

model_file_gen = 'out_models_RL/dcgan_model_gen_132.h5'
model_file_dis = 'out_models_RL/dcgan_model_dis_132.h5'
model_file_invg = 'out_models_invg_RL/dcgan_model_invg_50.h5'
#model_file_fl = 'out_models_FL_L_fromRL/model_best_RDF.h5'
#model_file_fl = 'out_models_FL_L_fromRL_add2/model_best_RDF.h5'
model_file_fl = 'out_models_FL_L_newdata_add/model_recent_RDF2.h5'

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        _, h, w = image.shape
        image = (image - 128)/128

        return image, label

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

def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

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
            l_img = L.Linear(3*128*128, 1, initialW=initializer),
            l_dis = L.Linear(512*8*8, 1, initialW=initializer),
            l_fdis = L.Linear(512*8*8, 1, initialW=initializer),
            l_FL = L.Linear(3, 1, initialW=initializer),
        )
        
    def __call__(self, img_error, dis_error, dis_output, test=False):
        h = F.reshape(F.sqrt(F.square(img_error)), (img_error.data.shape[0], 3*128*128))
        h = self.l_img(h)
        g = F.reshape(F.sqrt(F.square(dis_error)), (dis_error.data.shape[0], 512*8*8))
        g = self.l_dis(g)
        f = F.reshape(dis_output, (dis_output.data.shape[0], 512*8*8))
        f = self.l_fdis(f)
        ghf = F.sigmoid(self.l_FL(F.concat((h,g,f), axis=1)))
        return ghf

def test_dcgan_labeled(gen, invg, dis, fl, epoch0=0):
    Nall = 1000
    Nallt = 1000
    ind = range(Nall)
    indt = range(Nallt)

    #img_nn_n = np.zeros((100,3,128,128), dtype=np.float32)        
    #labeln = np.zeros((100,1), dtype=np.float32)
    img_nntL = np.zeros((100,3,128,128), dtype=np.float32)
    labeltL = np.zeros((100,1), dtype=np.float32) 
    pacc = 0.0
    plfl = 1.0

    for epoch in range(1):
        sum_l_invg = np.float32(0)
        
        countn_o = 0
        countp_o = 0     
        countn_l = 0
        countp_l = 0
        sum_L = 0.0
        for j in range(10):
            datatL = [testL.get_example(i + j*100) for i in range(batchsize)]
            #img_nntL = [datatL[i][0] for i in range(len(datatL))]
            for i in range(len(datatL)):
                labeltL[i] = datatL[i][1]
                img_nntL[i] = datatL[i][0]

            labelvt = Variable(cuda.to_gpu(labeltL))
            img_realt = Variable(cuda.to_gpu(img_nntL))

            with chainer.using_config('train', False), chainer.no_backprop_mode():
                img_gent = gen(invg(img_realt))
                dis_realt = dis(img_realt)
                dis_gent = dis(img_gent)
                outputt = fl(img_realt-img_gent, dis_realt-dis_gent, dis_realt)
            
            L_invgt = F.mean_squared_error(labelvt, outputt)
            sum_L += L_invgt.data

            #print  labelvt.data
            for ii in range(100):
                if outputt.data[ii] > 0.5 and labelvt.data[ii] == 1.0:
                    countp_o += 1.0
                if outputt.data[ii] < 0.5 and labelvt.data[ii] == 0.0:
                    countn_o += 1.0

            #print countp_o, countn_o
            for ii in range(100):
                if labelvt.data[ii] == 1:
                    countp_l += 1.0
                if labelvt.data[ii] == 0:
                    countn_l += 1.0
        accuracy = (countp_o + countn_o)/(countp_l + countn_l) 
        recall = countp_o/(countp_o + 500 - countp_o) 
        precision = countp_o/(countp_o + 500 - countn_o)
        acc_p = countp_o/500
        acc_n = countn_o/500  
        print epoch, accuracy, recall, precision, countp_l+countn_l, sum_L/10
        print acc_p, acc_n

xp = cuda.cupy
cuda.get_device(0).use()

gen = Generator()
dis = Discriminator()
invg = invG()
fl = FL()

gen.to_gpu()
dis.to_gpu()
invg.to_gpu()
fl.to_gpu()

serializers.load_hdf5(model_file_invg, invg)
serializers.load_hdf5(model_file_gen, gen)
serializers.load_hdf5(model_file_dis, dis)
serializers.load_hdf5(model_file_fl, fl)

testR = PreprocessedDataset('../dataset_stereo/dataset_test_Rz400.txt', '../dataset_stereo', True)
testL = PreprocessedDataset('../dataset_stereo/dataset_test_Lz400.txt', '../dataset_stereo', True)
#testR = PreprocessedDataset('../dataset_stereo/dataset_test_Ry400.txt', '../dataset_stereo', True)
#testL = PreprocessedDataset('../dataset_stereo/dataset_test_Ly400.txt', '../dataset_stereo', True)

try:
    os.mkdir(out_model_dir)
except:
    pass

test_dcgan_labeled(gen, invg, dis, fl)
