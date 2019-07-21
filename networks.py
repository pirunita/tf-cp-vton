from __future__ import absolute_import

import numpy as np
import tensorflow as tf

import network_base


DEFAULT_PADDING = 'SAME'


class FeatureCorrelation(network_base.BaseNetwork):
    def setup(self):
        b,h,w,c = self.inputs[0]
        # reshape features for matrix multiplication
        feature_A = tf.transpose(self.inputs[0], perm=[1, 3])
        feature_A = tf.reshape(feature_A, [b, c, h*w])
        feature_B = tf.reshape(self.inputs[1], [b, h*w, c])
        feature_mul = tf.matmul(feature_B, feature_A)
        correlation_tensor = tf.reshape(feature_mul, [b, h, w, h*w])
        return correlation_tensor

        

class FeatureExtraction(network_base.BaseNetwork):
    """ Extract feature module
    """
    def setup(self):
        (self.feed('preprocess')
             .conv(4, 4, self.inputs, self.ngf, 2, 2, name='downconv')
             .batch_normalization(name='batch_norm'))
        
        for i in range(self.n_layers):
            in_ngf = 2**i * self.ngf if 2**i * self.ngf < 512 else 512
            out_ngf = 2**(i+1) * self.ngf if 2**i * self.ngf < 512 else 512
            (self.feed('batch_norm')
                 .conv(4, 4, in_ngf, out_ngf, 2, 2, name='conv %d_1' % (i + 1))
                 .batch_normalization(name='batch_norm%d_1' % (i + 1)))
        
        (self.feed('batch_norm3_1')
             .conv(3, 3, 512, 512, 1, 1, name='conv4_1')
             .batch_norm(name='batch_norm4_1')
             .conv(3, 3, 512, 512, 1, 1, name='conv4_2'))


    def __init__(self):
        # The current list of terminal nodes
        self.ngf = 64
        self.n_layers = 3


class FeatureL2Norm(network_base.BaseNetwork):
    def setup(self):
        #expand_as
        epsilon = 1e-6
        norm = tf.expand_dims(self.pow(self.sum(self.pow(self.inputs,2), 1) + epsilon, 0.5), 1)
        return self.div(self.inputs, norm)

class FeatureRegression(network_base.BaseNetwork):

    def setup(self):
        (self.feed('FeatureRegression')
             .conv(4, 4, self.input_nc, 512, 2, 2, name='conv1_1', relu=False)
             .batch_normalization(name='batchnorm1_1')
             .relu(name='relu1_1')
             .conv(4, 4, 512, 256, 2, 2, name='conv2_1', relu=False)
             .batch_normalization(name='batchnorm2_1')
             .relu(name='relu2_1')
             .conv(4, 4, 256, 128, 2, 2, name='conv3_1', relu=False)
             .batch_normalization(name='batchnorm3_1')
             .relu(name='relu3_1')
             .conv(4, 4, 128, 64, 2, 2, name='conv4_1', relu=False)
             .batch_normalization(name='batchnorm4_1')
             .relu(name='relu4_1')

             #linear 추가해야함
             .tanh(name='tanh'))

    def __init__(self, input_nc=512, output_dim=6):
        self.input_nc = input_nc
    

class TpsGridGen(network_base.BaseNetwork):
    
    def setup(self):
        # apply_transformation
        theta = self.inputs[0]
        points = self.inputs[1]
        # points should be in the [B,H,W,2] format,
        # [:,:,:,0] are the X coords
        # [:,:,:,1] are the Y coords

        # input are the corresponding control points
        batch_size = tf.shape(theta)[0]
        # split theta into point coordinates
        Q_X = tf.squeeze(theta[:, :self.N, :, :], axis=3)
        Q_Y = tf.squeeze(theta[:, self.N:, :, :], axis=3)
        Q_X = Q_X + tf.broadcast_to(self.P_X_base, Q_X)
        Q_Y = Q_Y + tf.broadcast_to(self.P_Y_base, Q_Y)
        
        points_b = tf.shape(points)[0]
        points_h = tf.shape(points)[1]
        points_w = tf.shape(points)[2]

        P_X = tf.broadcast_to(self.P_X, [1, points_h, points_w, 1, self.N])
        P_Y = tf.broadcast_to(self.P_Y, [1, points_h, points_w, 1, self.N])

        # compute weights for non-linear part
        W_X = tf.matmul(tf.broadcast_to(self.Li[:,:self.N,:self.N], [batch_size, self.N, self.N]), Q_X)
        W_Y = tf.matmul(tf.broadcast_to(self.Li[:,:self.N,:self.N], [batch_size, self.N, self.N]), Q_Y)

        # reshape for W_X, W_Y: [B,H,W,1,N]

    def __init__(self,
                 theta,
                 points,
                 out_h=256, 
                 out_w=192, 
                 use_regular_grid=True, 
                 grid_size=3, 
                 reg_factor=0):
        self.theta = theta
        self.points = points
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X, grid_Y: size[1, H, W, 1, 1]
        self.grid_X = tf.convert_to_tensor(self.grid_X)
        self.grid_Y = tf.convert_to_tensor(self.grid_Y)
        
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size # 9
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))
            P_Y = np.reshape(P_Y, (-1, 1))
            P_X = tf.convert_to_tensor(P_X)
            P_Y = tf.convert_to_tensor(P_Y)
            self.P_X_base = P_X
            self.P_Y_base = P_Y
            self.Li = self.compute_L_inverse(P_X, P_Y)
            ## unsqueeze를 얼만큼 할 것인가?

    def compute_L_inverse(self, X, Y):
        N = tf.shape(X)[0]
        # construct matrix K
        Xmat = self.expand_dim(N, N, name='expand_dim1')
        Ymat = self.expand_dim(N, N, name='expand_dim2')
        XmatT = tf.transpose(Xmat, perm=[0, 1])
        YmatT = tf.transpose(Ymat, perm=[0, 1])
        P_dist_squared = tf.pow(XmatT, 2) + tf.pow(YmatT, 2)
        # make diagonal 1 to avoid NaN in log computation
        temp_P_dist_squared = P_dist_squared.eval()
        for i, v in np.ndenumerate(temp_P_dist_squared):
            if v == 0:
                temp_P_dist_squared[i[0], i[1]] = 1
        
        P_dist_squared = tf.convert_to_tensor(P_dist_squared)
        K = tf.matmul(P_dist_squared, tf.math.log(P_dist_squared))
        #construct matrix L
        O = np.ones([N, 1])
        Z = np.ones([3, 3])
        O = tf.convert_to_tensor(O)
        Z = tf.convert_to_tensor(Z)
        P = tf.concat([O, X, Y], axis=1)
        L = tf.concat([tf.concat([K, P], axis=1), tf.concat([tf.transpose(P, perm=[0, 1]), Z], axis=1)], axis=0)
        Li = tf.matrix_inverse(L)
        return Li

class GMM(object):
    """ Geometric Matching Module
    """
    def __init__(self, args):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(22, ngf=64, n_layers=3)
        self.extractionB = FeatureExtraction(3, ngf=64, n_layers=3)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression()
        self.gridGen = TpsGridGen()

       
    