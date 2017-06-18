#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os

"""Adversarially Learned Inference
Page 14: CelebA model hyperparameters
Optimizer Adam (α = 10−4, β1 = 0.5)
Batch size 100 Epochs 123
Leaky ReLU slope 0.02
Weight, bias initialization Isotropic gaussian (µ = 0, σ = 0.01), Constant(0)
"""
batch_size = 64

z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb


def generator(input_z, input_txt=None, is_train=True, reuse=False, batch_size=batch_size):
    """ G(z) or G(z, RNN(txt)) / output (64, 64, 3) """
    s = image_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if input_txt is not None:
            net_txt = InputLayer(input_txt, name='g_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                    act=lrelu,
                    W_init = w_init, b_init=None, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')

        net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
                W_init=w_init, b_init=None, name='g_h0/dense')
        net_h0 = BatchNormLayer(net_h0, #act=tf.identity,
                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')

        net = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
        net = BatchNormLayer(net, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm')
        net = Conv2d(net, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
        net = BatchNormLayer(net, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm2')
        net = Conv2d(net, gf_dim*8, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
        net = BatchNormLayer(net, # act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
        net_h1 = ElementwiseLayer(layer=[net_h0, net], combine_fn=tf.add, name='g_h1_res/add')
        net_h1.outputs = lrelu(net_h1.outputs)

        net_h2 = DeConv2d(net_h1, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=None, name='g_h2/decon2d')
        # net_h2 = UpSampling2dLayer(net_h1, size=[s8, s8], is_scale=False, method=1,
        #         align_corners=False, name='g_h2/upsample2d')
        # net_h2 = Conv2d(net_h2, gf_dim*4, (3, 3), (1, 1),
        #         padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,# act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

        net = Conv2d(net_h2, gf_dim, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
        net = BatchNormLayer(net, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm')
        net = Conv2d(net, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
        net = BatchNormLayer(net, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm2')
        net = Conv2d(net, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
        net = BatchNormLayer(net, #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
        net_h3 = ElementwiseLayer(layer=[net_h2, net], combine_fn=tf.add, name='g_h3_res/add')
        net_h3.outputs = lrelu(net_h3.outputs)

        net_h4 = DeConv2d(net_h3, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=None, name='g_h4/decon2d')
        # net_h4 = UpSampling2dLayer(net_h3, size=[s4, s4], is_scale=False, method=1,
        #         align_corners=False, name='g_h4/upsample2d')
        # net_h4 = Conv2d(net_h4, gf_dim*2, (3, 3), (1, 1),
        #         padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=lrelu,#tf.nn.relu,#lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        net_h5 = DeConv2d(net_h4, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=None, name='g_h5/decon2d')
        # net_h5 = UpSampling2dLayer(net_h4, size=[s2, s2], is_scale=False, method=1,
        #         align_corners=False, name='g_h5/upsample2d')
        # net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
        #         padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

        net_ho = DeConv2d(net_h5, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        # net_ho = UpSampling2dLayer(net_h9, size=[s, s], is_scale=False, method=1,
        #         align_corners=False, name='g_ho/upsample2d')
        # net_ho = Conv2d(net_ho, c_dim, (3, 3), (1, 1),
        #         padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho, logits

def encoder_simple(input_images, input_txt=None, is_train=True, reuse=False):
    """ E(x) input (64, 64, 3), output z """
    s = image_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    # 32 16 8 4
    w_init = tf.random_normal_initializer(stddev=0.01)
    gamma_init = tf.random_normal_initializer(1., 0.01)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='ig_inputz')
        # print(net_in.outputs)
        # exit()
        net_h0 = Conv2d(net_in, df_dim, (2, 2), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='ig_h0/conv2d')
        net_h0 = BatchNormLayer(net_h0, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h0/batchnorm')

        net_h1 = Conv2d(net_h0, df_dim*2, (7, 7), (2, 2), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='ig_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h1/batchnorm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='ig_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h2/batchnorm')

        net_h3 = Conv2d(net_h2, df_dim*4, (7, 7), (2, 2), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='ig_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h3/batchnorm')

        net_h4 = Conv2d(net_h3, df_dim*8, (4, 4), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='ig_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h4/batchnorm')

        if input_txt is not None:
            net_txt = InputLayer(input_txt, name='ig_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, b_init=None, name='ig_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='ig_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='ig_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='ig_txt/tile')
            net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='ig_txt/concat')
            net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
                   padding='SAME', W_init=w_init, b_init=None, name='ig_txt/conv2d_2')
            net_h4 = BatchNormLayer(net_h4, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
                   is_train=is_train, gamma_init=gamma_init, name='ig_txt/batch_norm_2')
        # print(net_h4.outputs) # (100, 8, 8, 512)
        # exit()
        # net_ho = Conv2d(net_h4, df_dim*8, (1, 1), (1, 1), act=None,
        #         padding='VALID', W_init=w_init, b_init=None, name='ig_ho/conv2d')  # DH
        # print(net_h4.outputs) # (100, 1, 1, 512)
        # exit()
        net_ho = FlattenLayer(net_h4, name='ig_ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=z_dim, name='ig_ho/dense')
        # print(net_ho.outputs)
        # exit()
        return net_ho

# def encoder(input_images, is_train=True, reuse=False):
#     """ E(x) input (64, 64, 3), output z """
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init=tf.random_normal_initializer(1., 0.02)
#     df_dim = 128#64
#     with tf.variable_scope("encoder", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         # (nc) x 64 x 64
#         net_in = InputLayer(input_images, name='ig_input/images')
#         net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
#                 padding='SAME', W_init=w_init, name='p_h0/conv2d')
#
#         net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='ig_h1/conv2d')
#         net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig_h1/batchnorm')
#         net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='ig_h2/conv2d')
#         net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig_h2/batchnorm')
#         net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='ig_h3/conv2d')
#         net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig_h3/batchnorm')
#
#         net_h = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
#                 padding='VALID', W_init=w_init, b_init=None, name='ig_h3/conv2d2')
#         net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig_h3/batchnorm2')
#         net_h = Conv2d(net_h, df_dim*2, (3, 3), (1, 1), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='ig_h3/conv2d3')
#         net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig_h3/batchnorm3')
#         net_h = Conv2d(net_h, df_dim*8, (3, 3), (1, 1), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='ig_h3/conv2d4')
#         net_h = BatchNormLayer(net_h, #act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig_h3/batchnorm4')
#         net_h3 = ElementwiseLayer(layer=[net_h3, net_h], combine_fn=tf.add, name='ig_h3/add')
#         net_h3.outputs = tl.act.lrelu(net_h3.outputs, 0.2)
#
#         net_h4 = Conv2d(net_h3, df_dim*2, (4, 4), (1, 1), padding='SAME',
#                 W_init=w_init, name='ig_h4/conv2d_2')
#         # print(net_h4.outputs)   # (100, 4, 4, 256)
#         # exit()
#         # 1 x 1 x 1
#         net_h4 = FlattenLayer(net_h4, name='ig_h4/flatten')
#         net_h4 = DenseLayer(net_h4, n_units=z_dim,
#                 act=tf.identity, W_init=w_init, b_init = None, name='ig/h4/embed')
#     return net_h4

def encoder_resnet(input_images, input_txt=None, is_train=True, reuse=False):
    """ E(x) 64x64 --> z """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    df_dim = 128#64
    with tf.variable_scope("encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        # (nc) x 64 x 64
        net_in = InputLayer(input_images, name='ig_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu,
                padding='SAME', W_init=w_init, name='p_h0/conv2d')
        # print(net_h0.outputs) # (100, 32, 32, 128)
        # exit()
        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='ig_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h1/batchnorm')
        net_h1 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='ig_h1/conv2d2')
        net_h1 = BatchNormLayer(net_h1, #act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h1/batchnorm2')
        # print(net_h1.outputs) # (100, 8, 8, 512)
        # exit()
        net = Conv2d(net_h1, df_dim*1, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='ig_h1_res/conv2d')
        net = BatchNormLayer(net, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h1_res/batchnorm')
        net = Conv2d(net, df_dim*1, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='ig_h1_res/conv2d2')
        net = BatchNormLayer(net, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='ig_h1_res/batchnorm2')
        net = Conv2d(net, df_dim*4, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='ig_h1_res/conv2d3')
        net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='ig_h1_res/batchnorm3')
        net_h1 = ElementwiseLayer(layer=[net_h1, net], combine_fn=tf.add, name='ig_h1_res/add')
        net_h1.outputs = lrelu(net_h1.outputs)

        # print(net_h1.outputs) # (100, 8, 8, 512)
        # exit()

        # net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
        #         padding='SAME', W_init=w_init, b_init=None, name='ig_h2/conv2d')
        # net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='ig_h2/batchnorm')
        # print(net_h2.outputs)
        # exit()

        net_h2 = Conv2d(net_h1, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='ig_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='ig_h2/batchnorm')
        # print(net_h3.outputs)
        # exit()

        net = Conv2d(net_h2, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='ig_h3_res/conv2d2')
        net = BatchNormLayer(net, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='ig_h3_res/batchnorm2')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='ig_h3_res/conv2d3')
        net = BatchNormLayer(net, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='ig_h3_res/batchnorm3')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='ig_h3_res/conv2d4')
        net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='ig_h3_res/batchnorm4')
        net_h3 = ElementwiseLayer(layer=[net_h2, net], combine_fn=tf.add, name='ig_h3_res/add')
        net_h3.outputs = lrelu(net_h3.outputs)
        # print(net_h3.outputs)
        # exit()
        if input_txt is not None:
            net_txt = InputLayer(input_txt, name='ig_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, b_init=None, name='ig_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='ig_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='ig_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='ig_txt/tile')
            net_h3_concat = ConcatLayer([net_h3, net_txt], concat_dim=3, name='ig_txt/concat')
            net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),
                   padding='SAME', W_init=w_init, b_init=None, name='ig_txt/conv2d_2')
            net_h3 = BatchNormLayer(net_h3, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
                   is_train=is_train, gamma_init=gamma_init, name='ig_txt/batch_norm_2')

        net_h3 = Conv2d(net_h3, df_dim*4, (4, 4), (1, 1), padding='SAME',
                W_init=w_init, name='ig_h3/conv2d_2')
        # print(net_h3.outputs)
        # exit()
        net_ho = FlattenLayer(net_h3, name='ig_ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=z_dim, act=tf.identity,
                W_init=w_init, b_init = None, name='ig/ho/embed')
    return net_ho

def discriminator_x(input_images, input_txt=None, is_train=True, reuse=False):
    """ D(x) input (64, 64, 3) """
    w_init = tf.random_normal_initializer(stddev=0.01)
    # w_init1 = tf.random_normal_initializer(stddev=0.01 * 2 * 2)
    # w_init2 = tf.random_normal_initializer(stddev=0.01 * 7 * 7)
    # w_init3 = tf.random_normal_initializer(stddev=0.01 * 5 * 5)
    # w_init4 = tf.random_normal_initializer(stddev=0.01 * 7 * 7)
    # w_init5 = tf.random_normal_initializer(stddev=0.01 * 4 * 4)
    gamma_init=tf.random_normal_initializer(1., 0.01)

    with tf.variable_scope("discriminator_x", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_images, name='dx_input/images')
        net_h0 = Conv2d(net_in, df_dim, (2, 2), (1, 1), act=lrelu,
                padding='VALID', W_init=w_init, name='dx_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (7, 7), (2, 2), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='dx_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='dx_h1/batchnorm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='dx_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='dx_h2/batchnorm')

        net_h3 = Conv2d(net_h2, df_dim*4, (7, 7), (2, 2), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='dx_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='dx_h3/batchnorm')

        net_h4 = Conv2d(net_h3, df_dim*8, (4, 4), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='dx_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=lrelu,
                is_train=is_train, gamma_init=gamma_init, name='dx_h4/batchnorm')

        if input_txt is not None:
            net_txt = InputLayer(input_txt, name='dx_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, b_init=None, name='dx_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='dx_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='dx_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='dx_txt/concat')
            net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
                   padding='SAME', W_init=w_init, b_init=None, name='dx_txt/conv2d_2')
            net_h4 = BatchNormLayer(net_h4, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
                   is_train=is_train, gamma_init=gamma_init, name='dx_txt/batch_norm_2')

        # print(net_h4.outputs, df_dim*8)
        # exit()
        net_ho = FlattenLayer(net_h4, name='dx_ho/flatten')
        # net_ho = DenseLayer(net_ho, n_units=z_dim, act=tf.identity,
        #         W_init = w_init, name='dx_ho/dense') # 512
        # print(net_ho.outputs)
        # exit()
        return net_ho

# def discriminator_x(input_images, input_txt=None, is_train=True, reuse=False): # cnn_encoder_resnet
#     """ D(x) or D(x, RNN(txt)) / x=(64, 64, 3), output z """
#     # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py  d_encode_image
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init=tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#
#     df_dim = 64
#     with tf.variable_scope("discriminator_x", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         # (nc) x 64 x 64
#         net_in = InputLayer(input_images, name='dx_input/images')
#
#         # net_in = DropoutLayer(net_in, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_in')
#         net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                 padding='SAME', W_init=w_init, name='dx_h0/conv2d')
#
#         # net_h0 = DropoutLayer(net_h0, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_h0')
#         net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h1/conv2d')
#         net_h1 = BatchNormLayer(net_h1, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h1/batchnorm')
#         # net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_h1')
#         net_h1 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h1/conv2d2')
#         net_h1 = BatchNormLayer(net_h1, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h1/batchnorm2')
#         # net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_h2')
#         net_h1 = Conv2d(net_h1, df_dim*8, (4, 4), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h1/conv2d3')
#         net_h1 = BatchNormLayer(net_h1, #act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h1/batchnorm3')
#
#         # net_h3 = DropoutLayer(net_h3, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_h3')
#         net_h = Conv2d(net_h1, df_dim*2, (1, 1), (1, 1), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h_res/conv2d2')
#         net_h = BatchNormLayer(net_h, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h_res/batchnorm2')
#         # net_h = DropoutLayer(net_h, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_resh1')
#         net_h = Conv2d(net_h, df_dim*2, (3, 3), (1, 1), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h_res/conv2d3')
#         net_h = BatchNormLayer(net_h, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h_res/batchnorm3')
#         # net_h = DropoutLayer(net_h, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_resh2')
#         net_h = Conv2d(net_h, df_dim*8, (3, 3), (1, 1), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h_res/conv2d4')
#         net_h = BatchNormLayer(net_h, #act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h_res/batchnorm4')
#         net_h2 = ElementwiseLayer(layer=[net_h1, net_h], combine_fn=tf.add, name='dx_h_res/add')
#         net_h2.outputs = tl.act.lrelu(net_h2.outputs, 0.2)
#
#         if input_txt is not None:
#             net_txt = InputLayer(input_txt, name='dx_input_txt')
#             net_txt = DenseLayer(net_txt, n_units=t_dim, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                    W_init=w_init, b_init=None, name='dx_reduce_txt/dense')
#             net_txt = ExpandDimsLayer(net_txt, 1, name='dx_txt/expanddim1')
#             net_txt = ExpandDimsLayer(net_txt, 1, name='dx_txt/expanddim2')
#             net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
#             net_h2_concat = ConcatLayer([net_h2, net_txt], concat_dim=3, name='dx_txt/concat')
#             net_h2 = Conv2d(net_h2_concat, df_dim*8, (1, 1), (1, 1),
#                    padding='SAME', W_init=w_init, b_init=None, name='dx_txt/conv2d_2')
#             net_h2 = BatchNormLayer(net_h2, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                    is_train=is_train, gamma_init=gamma_init, name='dx_txt/batch_norm_2')
#
#         # net_h2 = DropoutLayer(net_h2, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_after_concat')
#         net_ho = Conv2d(net_h2, df_dim*2, (4, 4), (1, 1), padding='SAME',
#                 W_init=w_init, name='dx_ho/conv2d_2')
#         # print(net_ho.outputs)   # (100, 4, 4, 128)
#         # 1 x 1 x 1
#         # net_ho = DropoutLayer(net_ho, keep=0.8, is_fix=True, is_train=is_train, name='dx_drop_ho')
#         net_ho = FlattenLayer(net_ho, name='dx_ho/flatten')
#         # exit()
#         net_ho = DenseLayer(net_ho, n_units=z_dim, act=tf.identity,
#                 W_init=w_init, b_init = None, name='dx/h4/embed')
#     return net_ho

def discriminator_z(input_z, is_train=True, reuse=False):
    """ D(z) input z """
    w_init = tf.random_normal_initializer(stddev=0.01)
    lrelu = lambda x: tl.act.lrelu(x, 0.02)

    with tf.variable_scope("discriminator_z", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='dz_input/z')
        # net_in = ReshapeLayer(net_in, [-1, 1, 1, z_dim], name='dz_reshape')

        net_in = DropoutLayer(net_in, keep=0.8, is_fix=True, is_train=is_train, name='dz_in/drop')
            # net_h0 = Conv2d(net_in, 1024, (1, 1), (1, 1), act=lrelu,
            #         padding='VALID', W_init=w_init, name='dz_h0/conv2d')
        net_h0 = DenseLayer(net_in, n_units=1024, act=lrelu,
                W_init=w_init, name='dz_h0/conv2d')

        net_h0 = DropoutLayer(net_h0, keep=0.8, is_fix=True, is_train=is_train, name='dz_h0/drop')
            # net_h1 = Conv2d(net_h0, 1024, (1, 1), (1, 1), act=lrelu,
            #         padding='VALID', W_init=w_init, name='dz_h1/conv2d')
        net_h1 = DenseLayer(net_h0, n_units=1024, act=lrelu,
                W_init=w_init, name='dz_h1/conv2d')

        # net_h1 = FlattenLayer(net_h1, name='dz_flatten')
        # print(net_h1.outputs) # 1024
        # exit()
        return net_h1

def discriminator_combine_xz(x, z, is_train=True, reuse=False):
    """ combine D(x) or D(x, RNN(txt)) with D(z), output real/fake """
    w_init = tf.random_normal_initializer(stddev=0.01)
    lrelu = lambda x: tl.act.lrelu(x, 0.02)

    with tf.variable_scope("discriminator_xz", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in_x = InputLayer(x, name='d_input/x')
        net_in_z = InputLayer(z, name='d_input/z')
        net_in = ConcatLayer([net_in_z, net_in_x], concat_dim=1, name='d/concat')
        # print(net_in.outputs)
        # exit()
        # net_in = ExpandDimsLayer(net_in, 1 , name='d/expanddim1')
        # net_in = ExpandDimsLayer(net_in, 1 , name='d/expanddim2')

        net_in = DropoutLayer(net_in, keep=0.8, is_fix=True, is_train=is_train, name='d_in/drop')
        # net_h0 = Conv2d(net_in, 2048, (1, 1), (1, 1), act=lrelu,
        #         padding='VALID', W_init=w_init, name='d_h0/conv2d')
        net_h0 = DenseLayer(net_in, n_units=1024,#2048,
                act=lrelu,
                W_init=w_init, name='d_h0/conv2d')

        net_h0 = DropoutLayer(net_h0, keep=0.8, is_fix=True, is_train=is_train, name='d_h0/drop')
        # net_h1 = Conv2d(net_h0, 2048, (1, 1), (1, 1), act=lrelu,
        #         padding='VALID', W_init=w_init, name='d_h1/conv2d')
        net_h1 = DenseLayer(net_h0, n_units=1024,#2048,
                act=lrelu,
                W_init=w_init, name='d_h1/conv2d')

        net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, is_train=is_train, name='d_h1/drop')
        # net_ho = Conv2d(net_h1, 1, (1, 1), (1, 1), act=None,
        #         padding='VALID', W_init=w_init, name='d_ho/conv2d')
        net_ho = DenseLayer(net_h1, n_units=1, act=tf.identity,
                W_init=w_init, name='d_ho/conv2d')
        # print(net_ho.outputs) # 1
        # exit()
        # net_ho = FlattenLayer(net_ho, name='d_ho/flatten')
        # print(net_ho.outputs) # 1
        # exit()
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
        return net_ho, logits

def discriminator(x, z, input_txt=None, is_train=True, reuse=False):
    """ D(x, z) or D(x, z, text)
    x=64x64
    """
    net_z = discriminator_z(z, is_train=is_train, reuse=reuse)
    net_x = discriminator_x(x, input_txt=input_txt, is_train=is_train, reuse=reuse)
    net_d, logits = discriminator_combine_xz(net_x.outputs, net_z.outputs, is_train=is_train, reuse=reuse)
    net_d.all_params.extend(net_x.all_params)
    net_d.all_params.extend(net_z.all_params)
    return net_d, logits

## follow DCGAN architecture / WORK but no deep enough for flower dataset
# def generator(input_z, input_txt=None, is_train=True, reuse=False, batch_size=batch_size):
#     """ G(z) input z, output (64, 64, 3) """
#     s = image_size
#     s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
#     # 32 16 8 4
#     w_init = tf.random_normal_initializer(stddev=0.01)
#     gamma_init = tf.random_normal_initializer(1., 0.01)
#     gf_dim = 128
#
#     with tf.variable_scope("generator", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         net_in = InputLayer(input_z, name='g_inputz')
#
#         if input_txt is not None:
#             net_txt = InputLayer(input_txt, name='g_input_txt')
#             net_txt = DenseLayer(net_txt, n_units=t_dim,
#                     act=lambda x: tl.act.lrelu(x, 0.2),
#                     W_init = w_init, b_init=None, name='g_reduce_text/dense')
#             # paper 4.1 : and then concatenated to the noise vector z
#             net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')
#
#         net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
#                 W_init=w_init, b_init=None, name='g_h0/dense')
#         net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')
#         net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
#                 gamma_init=gamma_init, name='g_h0/batch_norm')
#
#         net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
#                 padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=None, name='g_h1/decon2d')
#         net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
#                 gamma_init=gamma_init, name='g_h1/batch_norm')
#
#         net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
#                 padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=None, name='g_h2/decon2d')
#         net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
#                 gamma_init=gamma_init, name='g_h2/batch_norm')
#
#         net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
#                 padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=None, name='g_h3/decon2d')
#         net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
#                 gamma_init=gamma_init, name='g_h3/batch_norm')
#
#         net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(s, s), strides=(2, 2),
#                 padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_h4/decon2d')
#         logits = net_h4.outputs
#         net_h4.outputs = tf.nn.tanh(net_h4.outputs)
#     return net_h4, logits
#
# def encoder(input_images, input_txt=None, is_train=True, reuse=False):
#     """ E(x) input (64, 64, 3), output z """
#     s = image_size
#     s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
#     # 32 16 8 4
#     w_init = tf.random_normal_initializer(stddev=0.01)
#     gamma_init = tf.random_normal_initializer(1., 0.01)
#     df_dim = 128
#
#     with tf.variable_scope("encoder", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         net_in = InputLayer(input_images, name='ig_inputz')
#
#         net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
#                 padding='SAME', W_init=w_init, name='ig/h0/conv2d')
#
#         net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='ig/h1/conv2d')
#         net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig/h1/batch_norm')
#
#         # if name != 'cnn': # debug for training image encoder in step 2
#         #     net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, name='p/h1/drop')
#
#         net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='ig/h2/conv2d')
#         net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig/h2/batch_norm')
#
#         # if name != 'cnn': # debug for training image encoder in step 2
#         #     net_h2 = DropoutLayer(net_h2, keep=0.8, is_fix=True, name='p/h2/drop')
#
#         net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='ig/h3/conv2d')
#         net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='ig/h3/batch_norm')
#
#         # if name != 'cnn': # debug for training image encoder in step 2
#         #     net_h3 = DropoutLayer(net_h3, keep=0.8, is_fix=True, name='p/h3/drop')
#         # print(net_h3.outputs)
#         # exit()
#
#         if input_txt is not None:
#             net_txt = InputLayer(input_txt, name='ig_input_txt')
#             net_txt = DenseLayer(net_txt, n_units=t_dim, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                    W_init=w_init, b_init=None, name='ig_reduce_txt/dense')
#             net_txt = ExpandDimsLayer(net_txt, 1, name='ig_txt/expanddim1')
#             net_txt = ExpandDimsLayer(net_txt, 1, name='ig_txt/expanddim2')
#             net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='ig_txt/tile')
#             net_h3_concat = ConcatLayer([net_h3, net_txt], concat_dim=3, name='ig_txt/concat')
#             net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),
#                    padding='SAME', W_init=w_init, b_init=None, name='ig_txt/conv2d_2')
#             net_h3 = BatchNormLayer(net_h3, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                    is_train=is_train, gamma_init=gamma_init, name='ig_txt/batch_norm_2')
#
#         net_h4 = FlattenLayer(net_h3, name='ig/h4/flatten')
#         net_h4 = DenseLayer(net_h4, n_units=z_dim, act=tf.identity,
#                 W_init = w_init, b_init = None, name='ig/h4/embed')
#
#         ## DH add
#         # print("WARNING: FORCE ENCODER OUTPUT GAUSSIAN DISTRIBUTION !")
#         # mean, var = tf.nn.moments(net_h4.outputs, axes=[1])
#         # mean = tf.expand_dims(mean, 1)
#         # var = tf.expand_dims(var, 1)
#         # net_h4.outputs = (net_h4.outputs - mean) / tf.sqrt(var)
#     return net_h4
#
# def discriminator_x(input_images, input_txt=None, is_train=True, reuse=False):
#     """ D(x) input (64, 64, 3) """
#     w_init = tf.random_normal_initializer(stddev=0.01)
#     gamma_init=tf.random_normal_initializer(1., 0.01)
#     df_dim = 64
#
#     with tf.variable_scope("discriminator_x", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#
#         net_in = InputLayer(input_images, name='dx_input/images')
#         net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
#                 padding='SAME', W_init=w_init, name='dx_h0/conv2d')  # (64, 32, 32, 64)
#
#         net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h1/conv2d')
#         net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h1/batchnorm') # (64, 16, 16, 128)
#
#         net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h2/conv2d')
#         net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h2/batchnorm')    # (64, 8, 8, 256)
#
#         net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=None, name='dx_h3/conv2d')
#         net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='dx_h3/batchnorm') # (64, 4, 4, 512)  paper 4.1: when the spatial dim of the D is 4x4, we replicate the description embedding spatially and perform a depth concatenation
#
#         if input_txt is not None:
#             net_txt = InputLayer(input_txt, name='dx_input_txt')
#             net_txt = DenseLayer(net_txt, n_units=t_dim, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                    W_init=w_init, b_init=None, name='dx_reduce_txt/dense')
#             net_txt = ExpandDimsLayer(net_txt, 1, name='dx_txt/expanddim1')
#             net_txt = ExpandDimsLayer(net_txt, 1, name='dx_txt/expanddim2')
#             net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
#             net_h3_concat = ConcatLayer([net_h3, net_txt], concat_dim=3, name='dx_txt/concat')
#             net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),
#                    padding='SAME', W_init=w_init, b_init=None, name='dx_txt/conv2d_2')
#             net_h3 = BatchNormLayer(net_h3, act=lrelu,#lambda x: tl.act.lrelu(x, 0.2),
#                    is_train=is_train, gamma_init=gamma_init, name='dx_txt/batch_norm_2')
#
#         net_h4 = FlattenLayer(net_h3, name='dx_h4/flatten')          # (64, 8192)
#
#         net_h4 = DenseLayer(net_h4, n_units=512, act=tf.identity,
#                 W_init = w_init, name='dx_h4/dense')
#         # print(net_h4.outputs)
#         # exit()
#         # net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
#         #         W_init = w_init, name='d_h4/dense')
#         # logits = net_h4.outputs
#         # net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)  # (64, 1)
#     return net_h4
#
# def discriminator_z(input_z, is_train=True, reuse=False):
#     """ D(z) input z """
#     w_init = tf.random_normal_initializer(stddev=0.01)
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#     with tf.variable_scope("discriminator_z", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         net_in = InputLayer(input_z, name='dz_input/z')
#         # net_in = ReshapeLayer(net_in, [-1, 1, 1, z_dim], name='dz_reshape')
#
#         # if is_train:
#         #     net_in = DropoutLayer(net_in, keep=0.8, is_fix=True, name='dz_in/drop')
#         # net_h0 = Conv2d(net_in, 1024, (1, 1), (1, 1), act=lrelu,
#         #         padding='VALID', W_init=w_init, name='dz_h0/conv2d')
#         net_h0 = DenseLayer(net_in, n_units=512, act=lrelu, W_init=w_init, name='dz_h0/conv2d')
#
#         # if is_train:
#         #     net_h0 = DropoutLayer(net_h0, keep=0.8, is_fix=True, name='dz_h0/drop')
#         # net_h1 = Conv2d(net_h0, 1024, (1, 1), (1, 1), act=lrelu,
#         #         padding='VALID', W_init=w_init, name='dz_h1/conv2d')
#         net_h1 = DenseLayer(net_h0, n_units=512, act=lrelu, W_init=w_init, name='dz_h1/conv2d')
#
#         # net_h1 = FlattenLayer(net_h1, name='dz_flatten')
#         # print(net_h1.outputs) # 512
#         # exit()
#         return net_h1
#
# def discriminator_combine_xz(x, z, is_train=True, reuse=False):
#     """ input D(x), D(z), output real/fake """
#     w_init = tf.random_normal_initializer(stddev=0.01)
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#
#     with tf.variable_scope("discriminator", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         net_in_x = InputLayer(x, name='d_input/x')
#         net_in_z = InputLayer(z, name='d_input/z')
#         net_in = ConcatLayer([net_in_z, net_in_x], concat_dim=1, name='d/concat')
#         # print(net_in.outputs)
#         # exit()
#         # net_in = ExpandDimsLayer(net_in, 1 , name='d/expanddim1')
#         # net_in = ExpandDimsLayer(net_in, 1 , name='d/expanddim2')
#
#         # if is_train:
#         #     net_in = DropoutLayer(net_in, keep=0.8, is_fix=True, name='d_in/drop')
#         # net_h0 = Conv2d(net_in, 2048, (1, 1), (1, 1), act=lrelu,
#         #         padding='VALID', W_init=w_init, name='d_h0/conv2d')
#         net_h0 = DenseLayer(net_in, n_units=1024, act=lrelu, W_init=w_init, name='d_h0/conv2d')
#
#         # if is_train:
#         #     net_h0 = DropoutLayer(net_h0, keep=0.8, is_fix=True, name='d_h0/drop')
#         # net_h1 = Conv2d(net_h0, 2048, (1, 1), (1, 1), act=lrelu,
#         #         padding='VALID', W_init=w_init, name='d_h1/conv2d')
#         net_h1 = DenseLayer(net_h0, n_units=1024, act=lrelu, W_init=w_init, name='d_h1/conv2d')
#
#         # if is_train:
#         #     net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, name='d_h1/drop')
#         # net_ho = Conv2d(net_h1, 1, (1, 1), (1, 1), act=None,
#         #         padding='VALID', W_init=w_init, name='d_ho/conv2d')
#         net_ho = DenseLayer(net_h1, n_units=1, act=tf.identity,#lrelu,
#                 W_init=w_init, name='d_ho/conv2d')
#         # print(net_ho.outputs) # 1
#         # exit()
#         # net_ho = FlattenLayer(net_ho, name='d_ho/flatten')
#         # print(net_ho.outputs) # 1
#         # exit()
#         logits = net_ho.outputs
#         net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
#         return net_ho, logits
#
# def discriminator(x, z, is_train=True, reuse=False):
#     """ D(x, z) """
#     net_z = discriminator_z(z, is_train=is_train, reuse=reuse)
#     net_x = discriminator_x(x, is_train=is_train, reuse=reuse)
#     net_d, logits = discriminator_combine_xz(net_x.outputs, net_z.outputs, is_train=is_train, reuse=reuse)
#     net_d.all_params.extend(net_x.all_params)
#     net_d.all_params.extend(net_z.all_params)
#     return net_d, logits



## for text-to-image mapping ===================================================
t_dim = 128         # text feature dimension
rnn_hidden_size = t_dim
vocab_size = 8000
word_embedding_size = 256
keep_prob = 1.0

def rnn_embed(input_seqs, is_train=True, reuse=False, return_embed=False):
    """ txt --> t_dim """
    w_init = tf.random_normal_initializer(stddev=0.02)
    if tf.__version__ <= '0.12.1':
        LSTMCell = tf.nn.rnn_cell.LSTMCell
    else:
        LSTMCell = tf.contrib.rnn.BasicLSTMCell
    with tf.variable_scope("rnnftxt", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = EmbeddingInputlayer(
                     inputs = input_seqs,
                     vocabulary_size = vocab_size,
                     embedding_size = word_embedding_size,
                     E_init = w_init,
                     name = 'rnn/wordembed')
        network = DynamicRNNLayer(network,
                     cell_fn = LSTMCell,
                     cell_init_args = {'state_is_tuple' : True, 'reuse': reuse},  # for TF1.1, TF1.2 dont need to set reuse
                     n_hidden = rnn_hidden_size,
                     dropout = (keep_prob if is_train else None),
                     initializer = w_init,
                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
                     return_last = True,
                     name = 'rnn/dynamic')
        return network

def cnn_encoder(inputs, is_train=True, reuse=False, name='cnnftxt', return_h3=False):
    """ 64x64 --> t_dim, for text-image mapping """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(True)

        net_in = InputLayer(inputs, name='/in')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='cnnf/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h1/batch_norm')

        # if name != 'cnn': # debug for training image encoder in step 2
        #     net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, name='p/h1/drop')

        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h2/batch_norm')

        # if name != 'cnn': # debug for training image encoder in step 2
        #     net_h2 = DropoutLayer(net_h2, keep=0.8, is_fix=True, name='p/h2/drop')

        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h3/batch_norm')

        # if name != 'cnn': # debug for training image encoder in step 2
        #     net_h3 = DropoutLayer(net_h3, keep=0.8, is_fix=True, name='p/h3/drop')

        net_h4 = FlattenLayer(net_h3, name='cnnf/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units= (z_dim if name == 'z_encoder' else t_dim),
                act=tf.identity,
                W_init = w_init, b_init = None, name='cnnf/h4/embed')
    if return_h3:
        return net_h4, net_h3
    else:
        return net_h4


## simple g1, d1 ===============================================================
def generator_txt2img_simple(input_z, input_rnn_embed=None, is_train=True, reuse=False, batch_size=64):
    """ z + (txt) --> 64x64 """
    s = image_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    gf_dim = 128

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if input_rnn_embed is not None:
            net_txt = InputLayer(input_rnn_embed, name='g_rnn_embed_input')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init = w_init, b_init=None, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_seq')
        else:
            print("No text info will be used, i.e. normal DCGAN")

        net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
                W_init=w_init, b_init=b_init, name='g_h0/dense')
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2), # stackGI use (4, 4) https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_h4/decon2d')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits

def discriminator_txt2img_simple(input_images, input_rnn_embed=None, is_train=True, reuse=False):
    """ 64x64 + (txt) --> real/fake """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')

        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')

        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        if input_rnn_embed is not None:
            net_txt = InputLayer(input_rnn_embed, name='d_rnn_embed_input')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, b_init=None, name='d_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            net_h3_concat = ConcatLayer([net_h3, net_txt], concat_dim=3, name='d_h3_concat')
            # net_h3_concat = net_h3 # no text info
            net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d_2')
            net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                   is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')
        else:
            print("No text info will be used, i.e. normal DCGAN")

        net_h4 = FlattenLayer(net_h3, name='d_h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d_h4/dense')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits


## default g1, d1 ==============================================================
def generator_txt2img_resnet(input_z, t_txt=None, is_train=True, reuse=False, batch_size=batch_size):
    """ z + (txt) --> 64x64 """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    s = image_size # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='g_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')

        net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
                W_init=w_init, b_init=None, name='g_h0/dense')
        net_h0 = BatchNormLayer(net_h0,  #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')

        net = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1_res/batch_norm')
        net = Conv2d(net, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1_res/batch_norm2')
        net = Conv2d(net, gf_dim*8, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
        net = BatchNormLayer(net, # act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
        net_h1 = ElementwiseLayer(layer=[net_h0, net], combine_fn=tf.add, name='g_h1_res/add')
        net_h1.outputs = tf.nn.relu(net_h1.outputs)

        # Note: you can also use DeConv2d to replace UpSampling2dLayer and Conv2d
        # net_h2 = DeConv2d(net_h1, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
        net_h2 = UpSampling2dLayer(net_h1, size=[s8, s8], is_scale=False, method=1,
                align_corners=False, name='g_h2/upsample2d')
        net_h2 = Conv2d(net_h2, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,# act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

        net = Conv2d(net_h2, gf_dim, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3_res/batch_norm')
        net = Conv2d(net, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3_res/batch_norm2')
        net = Conv2d(net, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
        net = BatchNormLayer(net, #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
        net_h3 = ElementwiseLayer(layer=[net_h2, net], combine_fn=tf.add, name='g_h3/add')
        net_h3.outputs = tf.nn.relu(net_h3.outputs)

        # net_h4 = DeConv2d(net_h3, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h4/decon2d'),
        net_h4 = UpSampling2dLayer(net_h3, size=[s4, s4], is_scale=False, method=1,
                align_corners=False, name='g_h4/upsample2d')
        net_h4 = Conv2d(net_h4, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        # net_h5 = DeConv2d(net_h4, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h5/decon2d')
        net_h5 = UpSampling2dLayer(net_h4, size=[s2, s2], is_scale=False, method=1,
                align_corners=False, name='g_h5/upsample2d')
        net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

        # net_ho = DeConv2d(net_h5, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        net_ho = UpSampling2dLayer(net_h5, size=[s, s], is_scale=False, method=1,
                align_corners=False, name='g_ho/upsample2d')
        net_ho = Conv2d(net_ho, c_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho, logits

def discriminator_txt2img_resnet(input_images, t_txt=None, is_train=True, reuse=False):
    """ 64x64 + (txt) --> real/fake """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        net = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='d_h4_res/conv2d')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d2')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d3')
        net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')
        net_h4 = ElementwiseLayer(layer=[net_h3, net], combine_fn=tf.add, name='d_h4/add')
        net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='d_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, name='d_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='d_h3_concat')
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
                    padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')
            net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')

        net_ho = Conv2d(net_h4, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='d_ho/conv2d')
        # 1 x 1 x 1
        # net_ho = FlattenLayer(net_h4, name='d_ho/flatten')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
    return net_ho, logits

def z_encoder(input_images, is_train=True, reuse=False):
    """ 64x64 -> z """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    with tf.variable_scope("z_encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        net = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='d_h4_res/conv2d')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d2')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d3')
        net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')
        net_h4 = ElementwiseLayer(layer=[net_h3, net], combine_fn=tf.add, name='d_h4/add')
        net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

        net_ho = FlattenLayer(net_h4, name='d_ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=z_dim, act=tf.identity,
                W_init = w_init, name='d_ho/dense')

        # w_init = tf.random_normal_initializer(stddev=0.02)
        # b_init = None
        # gamma_init = tf.random_normal_initializer(1., 0.02)
        #
        # net_in = InputLayer(input_images, name='p/in')
        # net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
        #         padding='SAME', W_init=w_init, name='p/h0/conv2d')
        #
        # net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
        #         padding='SAME', W_init=w_init, b_init=b_init, name='p/h1/conv2d')
        # net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='p/h1/batch_norm')
        #
        # net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
        #         padding='SAME', W_init=w_init, b_init=b_init, name='p/h2/conv2d')
        # net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='p/h2/batch_norm')
        #
        # net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
        #         padding='SAME', W_init=w_init, b_init=b_init, name='p/h3/conv2d')
        # net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='p/h3/batch_norm')
        #
        # net_h4 = FlattenLayer(net_h3, name='p/h4/flatten')
        # net_ho = DenseLayer(net_h4, n_units=z_dim,
        #         act=tf.identity,
        #         # act=tf.nn.tanh,
        #         W_init = w_init, name='p/h4/output_real_fake')
    return net_ho
