import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

## GAN for text to img =========================================================
batch_size = 64
vocab_size = 8000
word_embedding_size = 256    # paper said 1024 char-CNN-RNN
rnn_hidden_size = 128#256
keep_prob = 1.0
z_dim = 100         # Noise dimension
t_dim = 128         # Text feature dimension # paper said 128
image_size = 64     # 64 x 64
c_dim = 3           # for rgb
gf_dim = 128        # Number of conv in the first layer generator
df_dim = 64         # Number of conv in the first layer discriminator

## shallow network for flower ==================================================
def rnn_embed(input_seqs, is_train, reuse, return_embed=True):
    """MY IMPLEMENTATION, same weights for the Word Embedding and RNN in the discriminator and generator.
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    # w_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope("rnn", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = EmbeddingInputlayer(
                     inputs = input_seqs,
                     vocabulary_size = vocab_size,
                     embedding_size = word_embedding_size,
                     E_init = w_init,
                     name = 'wordembed')
        network = DynamicRNNLayer(network,
                     cell_fn = tf.nn.rnn_cell.LSTMCell,
                     n_hidden = rnn_hidden_size,
                     dropout = (keep_prob if is_train else None),
                     initializer = w_init,
                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
                     return_last = True,
                     name = 'dynamic')
    if return_embed:
        with tf.variable_scope("rnn", reuse=reuse):
            net_embed = DenseLayer(network, n_units = t_dim,
                            act = tf.identity,# W_init = initializer,
                            b_init = None, name='hidden_state_embedding')
            return net_embed
    else:
        return network

def generator_txt2img(input_z, net_rnn_embed=None, is_train=True, reuse=False):
    # IMPLEMENTATION based on : https://github.com/paarthneekhara/text-to-image/blob/master/model.py
    s = image_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if net_rnn_embed is not None:
            # paper 4.1 : the discription embedding is first compressed using a FC layer to small dim (128), followed by leaky-Relu
            net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init = w_init, b_init=None, name='g_reduce_text/dense')
            # paper 4.1 : and then concatenated to the noise vector z
            net_in = ConcatLayer([net_in, net_reduced_text], concat_dim=1, name='g_concat_z_seq')
        else:
            print("No text info will be used, i.e. normal DCGAN")

        net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
                W_init=w_init, b_init=b_init, name='g_h0/dense')                  # (64, 8192)
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(s, s), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_h4/decon2d')
        logits = net_h4.outputs
        # net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)  # DCGAN uses tanh
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits

def discriminator_txt2img(input_images, net_rnn_embed=None, is_train=True, reuse=False):
    # IMPLEMENTATION based on : https://github.com/paarthneekhara/text-to-image/blob/master/model.py
    #       https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')  # (64, 32, 32, 64)

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm') # (64, 16, 16, 128)

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')    # (64, 8, 8, 256)

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm') # (64, 4, 4, 512)  paper 4.1: when the spatial dim of the D is 4x4, we replicate the description embedding spatially and perform a depth concatenation

        if net_rnn_embed is not None:
            # paper : reduce the dim of description embedding in (seperate) FC layer followed by rectification
            net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, b_init=None, name='d_reduce_txt/dense')
            # net_reduced_text = net_rnn_embed  # if reduce_txt in rnn_embed
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1)  # you can use ExpandDimsLayer and TileLayer instead
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 2)
            net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 4, 4, 1], name='d_tiled_embeddings')

            net_h3_concat = ConcatLayer([net_h3, net_reduced_text], concat_dim=3, name='d_h3_concat') # (64, 4, 4, 640)
            # net_h3_concat = net_h3 # no text info
            net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d_2')   # paper 4.1: perform 1x1 conv followed by rectification and a 4x4 conv to compute the final score from D
            net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                   is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2') # (64, 4, 4, 512)
        else:
            print("No text info will be used, i.e. normal DCGAN")

        net_h4 = FlattenLayer(net_h3, name='d_h4/flatten')          # (64, 8192)
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d_h4/dense')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)  # (64, 1)
    return net_h4, logits

def cnn_encoder(inputs, is_train, reuse, name="cnn"):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='p/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='p/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='p/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='p/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='p/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='p/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=(t_dim if name=="cnn" else z_dim),
                act=tf.identity,
                W_init = w_init,
                b_init = None,
                name='p/h4/embed')
    return net_h4

## deep network for flower =====================================================
def generator_txt2img_deep(input_z, net_rnn_embed=None, is_train=True, reuse=False, is_large=False):
    # Generator with ResNet : line 93 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    s = image_size # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128 #256 #400#256 #196 # 128 for flower, 196 for MSCOCO   # <- gen filters in first conv layer [196] https://github.com/reedscot/icml2016/blob/master/scripts/train_coco.sh

    w_init = tf.random_normal_initializer(stddev=0.02)  # 73
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02) # 74

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if net_rnn_embed is not None:
            net_rnn_embed = DenseLayer(net_rnn_embed, n_units=t_dim,            # 95, 96
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init=w_init, b_init=None, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_rnn_embed], concat_dim=1, name='g_concat_z_seq') # 103  (64, 356) - 100+256=356
        else:
            print("No text info is used, i.e. DCGAN")

        ## 105 Note: ppwwyyxx - SpatialFullConvolution on 1x1 input is equivalent to a dense layer.
        # net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,          # 106 netG:add(SpatialFullConvolution(opt.nz + opt.nt, ngf * 8, 4, 4))
        #         W_init=w_init, b_init=b_init, name='g_h0/dense')                               # (64, gf_dim*8*4*4)
        # net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')# 106  (64, 4, 4, gf_dim*8) = (ngf*8) x 4 x 4
        net_h0 = DenseLayer(net_in, gf_dim*(8*2)*int(s16/2)*int(s16/2), act=tf.identity,          # for deep
                W_init=w_init, b_init=b_init, name='g_h0/dense')
        net_h0 = ReshapeLayer(net_h0, [-1, int(s16/2), int(s16/2), gf_dim*(8*2)], name='g_h0/reshape')# for deep
        net_h0 = BatchNormLayer(net_h0,  act=tf.nn.relu,                       # 107 no relu
                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')

        net_h3 = DeConv2d(net_h0, gf_dim*8, (4, 4), out_size=(s16, s16), strides=(2, 2),# 142 (64, 8, 8, gf_dim*4) = (ngf*4) x 4 x 4
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu,                        # 143 no relu
                is_train=is_train, gamma_init=gamma_init, name='g_h3/batch_norm')
        # 109 resnet
        # net_h1 = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),                       # 112 (64, 4, 4, gf_dim*2)
        #         padding='VALID', act=None, W_init=w_init, b_init=b_init, name='g_h1/conv2d')
        # net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,      # 113
        #         gamma_init=gamma_init, name='g_h1/batch_norm')
        # net_h2 = Conv2d(net_h1, gf_dim*2, (3, 3), (1, 1),                       # 114
        #         padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h2/conv2d')
        # net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,      # 115
        #         gamma_init=gamma_init, name='g_h2/batch_norm')
        # net_h3 = Conv2d(net_h3, gf_dim*8, (3, 3), (1, 1),                       # 116
        #         padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h3/conv2d')
        # net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu,                       # 117 no relu
        #         is_train=is_train, gamma_init=gamma_init, name='g_h3/batch_norm')

        net_h4 = DeConv2d(net_h3, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),# 142 (64, 8, 8, gf_dim*4) = (ngf*4) x 4 x 4
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h4/decon2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,                        # 143 no relu
                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        # 148 resnet (ngf*4) x 8 x 8 = (64, 8, 8, 512)
        # net_h5 = Conv2d(net_h4, gf_dim, (1, 1), (1, 1),                         # 148
        #         padding='VALID', act=None, W_init=w_init, b_init=b_init, name='g_h5/conv2d')
        # net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu, is_train=is_train,      # 149
        #         gamma_init=gamma_init, name='g_h5/batch_norm')
        # net_h6 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),                         # 150
        #         padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h6/conv2d')
        # net_h6 = BatchNormLayer(net_h6, act=tf.nn.relu, is_train=is_train,      # 151
        #         gamma_init=gamma_init, name='g_h6/batch_norm')
        # net_h7 = Conv2d(net_h6, gf_dim*4, (3, 3), (1, 1),                       # 152
        #         padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h7/conv2d')
        # net_h7 = BatchNormLayer(net_h7, act=tf.nn.relu,                        # 153 no relu
        #         is_train=is_train, gamma_init=gamma_init, name='g_h7/batch_norm')

        net_h8 = DeConv2d(net_h4, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),# 178 (64, 16, 16, gf_dim*2) = (ngf*2) x 16 x 16
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h8/decon2d')
        net_h8 = BatchNormLayer(net_h8, act=tf.nn.relu, is_train=is_train,      # 179, 180
                gamma_init=gamma_init, name='g_h8/batch_norm')

        net_h9 = DeConv2d(net_h8, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),# 183 (64, 32, 32, gf_dim) = (ngf) x 32 x 32
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h9/decon2d')
        net_h9 = BatchNormLayer(net_h9, act=tf.nn.relu, is_train=is_train,      # 184
                gamma_init=gamma_init, name='g_h9/batch_norm')

        net_ho = DeConv2d(net_h9, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),# 187 (64, 64, 64, 3)
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)                             # 188
    return net_ho, logits

def discriminator_txt2img_deep(input_images, net_rnn_embed=None, is_train=True, reuse=False):
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)  # 73
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)   # 74
    df_dim = 64  # 64 for flower, 196 for MSCOCO       # number of conv in the first layer discriminator [196] https://github.com/reedscot/icml2016/blob/master/scripts/train_coco.sh

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        # (nc) x 64 x 64
        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), # 199
                padding='SAME', W_init=w_init, name='d_h0/conv2d')              # (64, 32, 32, 64) = (ndf) x 32 x 32

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,             # 203  (64, 16, 16, df_dim*2) = (ndf*2) x 16 x 16
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),     # 204, 205
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,             # 208  (64, 8, 8, df_dim*4) = (ndf*4) x 8 x 8
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),     # 209, 210
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,             # 213  (64, 4, 4, df_dim*8) = (ndf*8) x 4 x 4
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),    # 214, no lrelu
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')
        # print(net_h3.outputs)
        net_h3 = Conv2d(net_h3, df_dim*16, (4, 4), (2, 2), act=None,             # deep
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h3_2/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),    # deep
                is_train=is_train, gamma_init=gamma_init, name='d_h3_2/batchnorm')
        # net_h = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,              # 219
        #         padding='VALID', W_init=w_init, b_init=b_init, name='d_h3/conv2d2')
        # net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),       # 220
        #         is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm2')
        # net_h = Conv2d(net_h, df_dim*2, (3, 3), (1, 1), act=None,               # 221
        #         padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d3')
        # net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),       # 222
        #         is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm3')
        # net_h = Conv2d(net_h, df_dim*8, (3, 3), (1, 1), act=None,               # 223
        #         padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d4')
        # net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),      # 224
        #         is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm4')
        # net_h3 = net_h

        if net_rnn_embed is not None:  # 232
            net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,         # 233, 234
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, b_init=None, name='d_reduce_txt/dense')
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1) # 235
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 2) # 236
            # net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 4, 4, 1], name='d_tiled_embeddings')
            net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 2, 2, 1], name='d_tiled_embeddings')  # for deep
            # print(net_h3.outputs)
            # print(net_reduced_text.outputs)
            # exit()
            net_h3_concat = ConcatLayer([net_h3, net_reduced_text], concat_dim=3, name='d_h3_concat') # 242  if t_dim = 256 : (64, 4, 4, 786); if t_dim = 128 :(64, 4, 4, 640)
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),            # 244 (64, 4, 4, df_dim*8)
                    padding='VALID', W_init=w_init, b_init=b_init, name='d_h3/conv2d_2')
            net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2), # 245
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')
        else:
            print("No text info is used, i.e. DCGAN")
        # print(net_h3.outputs)
        # exit()
        # net_h4 = Conv2d(net_h3, 1, (4, 4), (1, 1), padding='VALID', W_init=w_init, name='d_h4/conv2d_2') # 246 (64, 1, 1, 1), if padding='SAME' (64, 4, 4, 1)
        net_h4 = Conv2d(net_h3, 1, (2, 2), (1, 1), padding='VALID', W_init=w_init, name='d_h4/conv2d_2') # for deep
        # 1 x 1 x 1
        net_h4 = FlattenLayer(net_h4, name='d_h4/flatten')                      # 249 (64, 1)
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits

def cnn_encoder_deep(input_images, is_train=True, reuse=False, name='cnn'):
    w_init = tf.random_normal_initializer(stddev=0.02)  # 73
    b_init = None
    gamma_init=tf.random_normal_initializer(1., 0.02)   # 74
    df_dim = 64   #  64 for flower, 196 for MSCOCO      # number of conv in the first layer discriminator [196] https://github.com/reedscot/icml2016/blob/master/scripts/train_coco.sh

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        # (nc) x 64 x 64
        net_in = InputLayer(input_images, name='p_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), # 199
                padding='SAME', W_init=w_init, name='p_h0/conv2d')              # (64, 32, 32, 64) = (ndf) x 32 x 32

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,             # 203  (64, 16, 16, df_dim*2) = (ndf*2) x 16 x 16
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),     # 204, 205
                is_train=is_train, gamma_init=gamma_init, name='p_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,             # 208  (64, 8, 8, df_dim*4) = (ndf*4) x 8 x 8
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),     # 209, 210
                is_train=is_train, gamma_init=gamma_init, name='p_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,             # 213  (64, 4, 4, df_dim*8) = (ndf*8) x 4 x 4
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),    # 214, no lrelu
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm')

        net_h3 = Conv2d(net_h3, df_dim*16, (4, 4), (2, 2), act=None,             # deep
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h3_2/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),    # deep
                is_train=is_train, gamma_init=gamma_init, name='p_h3_2/batchnorm')
        # 216 resnet
        # net_h = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,              # 219
        #         padding='VALID', W_init=w_init, b_init=b_init, name='p_h3/conv2d2')
        # net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),       # 220
        #         is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm2')
        # net_h = Conv2d(net_h, df_dim*2, (3, 3), (1, 1), act=None,               # 221
        #         padding='SAME', W_init=w_init, b_init=b_init, name='p_h3/conv2d3')
        # net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),       # 222
        #         is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm3')
        # net_h = Conv2d(net_h, df_dim*8, (3, 3), (1, 1), act=None,               # 223
        #         padding='SAME', W_init=w_init, b_init=b_init, name='p_h3/conv2d4')
        # net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),      # 224
        #         is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm4')
        # 228 end resnet

        net_h4 = Conv2d(net_h3, df_dim*2, (2, 2), (1, 1), padding='VALID', W_init=w_init, name='p_h4/conv2d_2')
        # net_h4 = Conv2d(net_h3, df_dim*2, (4, 4), (1, 1), padding='VALID', W_init=w_init, name='p_h4/conv2d_2')

        net_h4 = FlattenLayer(net_h4, name='p_h4/flatten')
        # print(net_h4.outputs)
        net_h4 = DenseLayer(net_h4, n_units=(t_dim if name=="cnn" else z_dim),
            act=tf.identity,
            W_init = w_init,
            b_init = None,
            name='p/h4/embed')
        # logits = net_h4.outputs
        # net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4

## large network for MSCOCO ====================================================
def generator_txt2img_resnet(input_z, net_rnn_embed=None, is_train=True, reuse=False, is_large=False):
    # Generator with ResNet : line 93 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    s = image_size # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128 #256 #400#256 #196 # 128 for flower, 196 for MSCOCO   # <- gen filters in first conv layer [196] https://github.com/reedscot/icml2016/blob/master/scripts/train_coco.sh

    w_init = tf.random_normal_initializer(stddev=0.02)  # 73
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02) # 74

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if net_rnn_embed is not None:
            net_rnn_embed = DenseLayer(net_rnn_embed, n_units=t_dim,            # 95, 96
                    act=lambda x: tl.act.lrelu(x, 0.2), W_init = w_init, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_rnn_embed], concat_dim=1, name='g_concat_z_seq') # 103  (64, 356) - 100+256=356
        else:
            print("No text info is used, i.e. DCGAN")

        ## 105 Note: ppwwyyxx - SpatialFullConvolution on 1x1 input is equivalent to a dense layer.
        net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,          # 106 netG:add(SpatialFullConvolution(opt.nz + opt.nt, ngf * 8, 4, 4))
                W_init=w_init, b_init=b_init, name='g_h0/dense')                               # (64, gf_dim*8*4*4)
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')# 106  (64, 4, 4, gf_dim*8) = (ngf*8) x 4 x 4
        net_h0 = BatchNormLayer(net_h0,  #act=tf.nn.relu,                       # 107 no relu
                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')

        # 109 resnet
        net_h1 = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),                       # 112 (64, 4, 4, gf_dim*2)
                padding='VALID', act=None, W_init=w_init, b_init=b_init, name='g_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,      # 113
                gamma_init=gamma_init, name='g_h1/batch_norm')
        net_h2 = Conv2d(net_h1, gf_dim*2, (3, 3), (1, 1),                       # 114
                padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,      # 115
                gamma_init=gamma_init, name='g_h2/batch_norm')
        net_h3 = Conv2d(net_h2, gf_dim*8, (3, 3), (1, 1),                       # 116
                padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, # act=tf.nn.relu,                       # 117 no relu
                is_train=is_train, gamma_init=gamma_init, name='g_h3/batch_norm')
        net_h3.outputs = net_h3.outputs + net_h0.outputs                        # 121 (64, 4, 4, gf_dim*8) = (ngf*8) x 4 x 4
        # 121 end resnet

        if is_large is True: # 123 resnet
            net_h = Conv2d(net_h3, gf_dim*2, (1, 1), (1, 1),                    # 127
                    padding='VALID', act=None, W_init=w_init, b_init=b_init, name='g_h3/conv2d2')
            net_h = BatchNormLayer(net_h, act=tf.nn.relu, is_train=is_train,    # 128
                    gamma_init=gamma_init, name='g_h3/batch_norm2')
            net_h = Conv2d(net_h, gf_dim*2, (3, 3), (1, 1),                     # 129
                    padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h3/conv2d3')
            net_h = BatchNormLayer(net_h, act=tf.nn.relu, is_train=is_train,    # 130
                    gamma_init=gamma_init, name='g_h3/batch_norm3')
            net_h = Conv2d(net_h, gf_dim*8, (3, 3), (1, 1),                     # 131
                    padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h3/conv2d4')
            net_h = BatchNormLayer(net_h, #act=tf.nn.relu,                      # 132
                    is_train=is_train, gamma_init=gamma_init, name='g_h3/batch_norm4')
            net_h3.outputs = net_h3.outputs + net_h.outputs                     # 136 (64, 4, 4, gf_dim*8) = (ngf*8) x 4 x 4
        net_h3.outputs = tf.nn.relu(net_h3.outputs)                             # 139

        net_h4 = DeConv2d(net_h3, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),# 142 (64, 8, 8, gf_dim*4) = (ngf*4) x 4 x 4
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h4/decon2d')
        net_h4 = BatchNormLayer(net_h4,# act=tf.nn.relu,                        # 143 no relu
                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        # 148 resnet (ngf*4) x 8 x 8 = (64, 8, 8, 512)
        net_h5 = Conv2d(net_h4, gf_dim, (1, 1), (1, 1),                         # 148
                padding='VALID', act=None, W_init=w_init, b_init=b_init, name='g_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu, is_train=is_train,      # 149
                gamma_init=gamma_init, name='g_h5/batch_norm')
        net_h6 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),                         # 150
                padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h6/conv2d')
        net_h6 = BatchNormLayer(net_h6, act=tf.nn.relu, is_train=is_train,      # 151
                gamma_init=gamma_init, name='g_h6/batch_norm')
        net_h7 = Conv2d(net_h6, gf_dim*4, (3, 3), (1, 1),                       # 152
                padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h7/conv2d')
        net_h7 = BatchNormLayer(net_h7, #act=tf.nn.relu,                        # 153 no relu
                is_train=is_train, gamma_init=gamma_init, name='g_h7/batch_norm')
        net_h7.outputs = net_h4.outputs + net_h7.outputs                        # 157 (64, 8, 8, gf_dim*4) = (ngf*4) x 8 x 8
        # 158 end resnet

        if is_large is True:# 159 resnet
            net_h = Conv2d(net_h7, gf_dim, (1, 1), (1, 1),                      # 163
                   padding='VALID', act=None, W_init=w_init, b_init=b_init, name='g_h7/conv2d1')
            net_h = BatchNormLayer(net_h, act=tf.nn.relu, is_train=is_train,    # 164
                   gamma_init=gamma_init, name='g_h7/batch_norm1')
            net_h = Conv2d(net_h, gf_dim, (3, 3), (1, 1),                       # 165
                   padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h7/conv2d2')
            net_h = BatchNormLayer(net_h, act=tf.nn.relu, is_train=is_train,    # 166
                   gamma_init=gamma_init, name='g_h7/batch_norm2')
            net_h = Conv2d(net_h, gf_dim*4, (3, 3), (1, 1),                     # 167
                   padding='SAME', act=None, W_init=w_init, b_init=b_init, name='g_h7/conv2d3')
            net_h = BatchNormLayer(net_h, #act=tf.nn.relu,                      # 168 no relu
                   is_train=is_train, gamma_init=gamma_init, name='g_h7/batch_norm3')
            net_h7.outputs = net_h.outputs + net_h7.outputs                     # 172
        net_h7.outputs = tf.nn.relu(net_h7.outputs)                             # 175 (64, 8, 8, gf_dim*4) = (ngf*4) x 8 x 8

        net_h8 = DeConv2d(net_h7, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),# 178 (64, 16, 16, gf_dim*2) = (ngf*2) x 16 x 16
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h8/decon2d')
        net_h8 = BatchNormLayer(net_h8, act=tf.nn.relu, is_train=is_train,      # 179, 180
                gamma_init=gamma_init, name='g_h8/batch_norm')

        net_h9 = DeConv2d(net_h8, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),# 183 (64, 32, 32, gf_dim) = (ngf) x 32 x 32
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h9/decon2d')
        net_h9 = BatchNormLayer(net_h9, act=tf.nn.relu, is_train=is_train,      # 184
                gamma_init=gamma_init, name='g_h9/batch_norm')

        net_ho = DeConv2d(net_h9, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),# 187 (64, 64, 64, 3)
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)                             # 188
    return net_ho, logits

def discriminator_txt2img_resnet(input_images, net_rnn_embed=None, is_train=True, reuse=False):
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)  # 73
    b_init = None
    gamma_init=tf.random_normal_initializer(1., 0.02)   # 74
    df_dim = 64  # 64 for flower, 196 for MSCOCO       # number of conv in the first layer discriminator [196] https://github.com/reedscot/icml2016/blob/master/scripts/train_coco.sh

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        # (nc) x 64 x 64
        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), # 199
                padding='SAME', W_init=w_init, name='d_h0/conv2d')              # (64, 32, 32, 64) = (ndf) x 32 x 32

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,             # 203  (64, 16, 16, df_dim*2) = (ndf*2) x 16 x 16
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),     # 204, 205
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,             # 208  (64, 8, 8, df_dim*4) = (ndf*4) x 8 x 8
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),     # 209, 210
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,             # 213  (64, 4, 4, df_dim*8) = (ndf*8) x 4 x 4
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),    # 214, no lrelu
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        # 216 resnet
        net_h = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,              # 219
                padding='VALID', W_init=w_init, b_init=b_init, name='d_h3/conv2d2')
        net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),       # 220
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm2')
        net_h = Conv2d(net_h, df_dim*2, (3, 3), (1, 1), act=None,               # 221
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d3')
        net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),       # 222
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm3')
        net_h = Conv2d(net_h, df_dim*8, (3, 3), (1, 1), act=None,               # 223
                padding='SAME', W_init=w_init, b_init=b_init, name='d_h3/conv2d4')
        net_h = BatchNormLayer(net_h, #act=lambda x: tl.act.lrelu(x, 0.2),      # 224
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm4')
        net_h3.outputs = tl.act.lrelu(net_h.outputs + net_h3.outputs, 0.2)      # 228, 230 (64, 4, 4, df_dim*8)
        # 228 end resnet

        if net_rnn_embed is not None:  # 232
            net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,         # 233, 234
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, name='d_reduce_txt/dense')
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1) # 235
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 2) # 236
            net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 4, 4, 1], name='d_tiled_embeddings')
            net_h3_concat = ConcatLayer([net_h3, net_reduced_text], concat_dim=3, name='d_h3_concat') # 242  if t_dim = 256 : (64, 4, 4, 786); if t_dim = 128 :(64, 4, 4, 640)
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),            # 244 (64, 4, 4, df_dim*8)
                    padding='VALID', W_init=w_init, b_init=b_init, name='d_h3/conv2d_2')
            net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2), # 245
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')
        else:
            print("No text info is used, i.e. DCGAN")
        net_h4 = Conv2d(net_h3, 1, (4, 4), (1, 1), padding='VALID', W_init=w_init, name='d_h4/conv2d_2') # 246 (64, 1, 1, 1), if padding='SAME' (64, 4, 4, 1)
        # 1 x 1 x 1
        net_h4 = FlattenLayer(net_h4, name='d_h4/flatten')                      # 249 (64, 1)
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits

def cnn_encoder_resnet(input_images, is_train=True, reuse=False, name='cnn'):
    w_init = tf.random_normal_initializer(stddev=0.02)  # 73
    b_init = None
    gamma_init=tf.random_normal_initializer(1., 0.02)   # 74
    df_dim = 64   #  64 for flower, 196 for MSCOCO      # number of conv in the first layer discriminator [196] https://github.com/reedscot/icml2016/blob/master/scripts/train_coco.sh

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        # (nc) x 64 x 64
        net_in = InputLayer(input_images, name='p_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), # 199
                padding='SAME', W_init=w_init, name='p_h0/conv2d')              # (64, 32, 32, 64) = (ndf) x 32 x 32

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,             # 203  (64, 16, 16, df_dim*2) = (ndf*2) x 16 x 16
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),     # 204, 205
                is_train=is_train, gamma_init=gamma_init, name='p_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,             # 208  (64, 8, 8, df_dim*4) = (ndf*4) x 8 x 8
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),     # 209, 210
                is_train=is_train, gamma_init=gamma_init, name='p_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,             # 213  (64, 4, 4, df_dim*8) = (ndf*8) x 4 x 4
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),    # 214, no lrelu
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm')

        # 216 resnet
        net_h = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,              # 219
                padding='VALID', W_init=w_init, b_init=b_init, name='p_h3/conv2d2')
        net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),       # 220
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm2')
        net_h = Conv2d(net_h, df_dim*2, (3, 3), (1, 1), act=None,               # 221
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h3/conv2d3')
        net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),       # 222
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm3')
        net_h = Conv2d(net_h, df_dim*8, (3, 3), (1, 1), act=None,               # 223
                padding='SAME', W_init=w_init, b_init=b_init, name='p_h3/conv2d4')
        net_h = BatchNormLayer(net_h, #act=lambda x: tl.act.lrelu(x, 0.2),      # 224
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm4')
        net_h3.outputs = tl.act.lrelu(net_h.outputs + net_h3.outputs, 0.2)      # 228, 230 (64, 4, 4, df_dim*8)
        # 228 end resnet

        net_h4 = Conv2d(net_h3, df_dim*2, (4, 4), (1, 1), padding='SAME', W_init=w_init, name='p_h4/conv2d_2')
        # 1 x 1 x 1
        net_h4 = FlattenLayer(net_h4, name='p_h4/flatten')                      # 249 (64, 1)
        net_h4 = DenseLayer(net_h4, n_units=(t_dim if name=="cnn" else z_dim),
            act=tf.identity,
            W_init = w_init,
            b_init = None,
            name='p/h4/embed')
        # logits = net_h4.outputs
        # net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4

## DCGAN =======================================================================
def generator_dcgan(inputs, net_rnn_embed=None, is_train=True, reuse=False):
    image_size = 64
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16)
    gf_dim = 64 # Dimension of gen filters in first conv layer. [64]
    c_dim = 3#FLAGS.c_dim # n_color 3
    batch_size = 64#FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim*8*s16*s16, W_init=w_init, b_init=b_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits

def discriminator_dcgan(inputs, net_rnn_embed=None, is_train=True, reuse=False):
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = 3#FLAGS.c_dim # n_color 3
    batch_size = 64#FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h4/lin_sigmoid')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits
