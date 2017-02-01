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

## simple network ==================================================
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

def generator_txt2img(input_z, net_rnn_embed=None, is_train=True, reuse=False, batch_size=64):
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

        net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2), # stackGI use (4, 4) https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
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

        if name != 'cnn': # debug for training image encoder in step 2
            net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, name='p/h1/drop')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='p/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h2/batch_norm')

        if name != 'cnn': # debug for training image encoder in step 2
            net_h2 = DropoutLayer(net_h2, keep=0.8, is_fix=True, name='p/h2/drop')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='p/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h3/batch_norm')

        if name != 'cnn': # debug for training image encoder in step 2
            net_h3 = DropoutLayer(net_h3, keep=0.8, is_fix=True, name='p/h3/drop')

        net_h4 = FlattenLayer(net_h3, name='p/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=(t_dim if name=="cnn" else z_dim),
                act=tf.identity,
                W_init = w_init,
                b_init = None,
                name='p/h4/embed')
    return net_h4

## large network ====================================================
def generator_txt2img_resnet(input_z, net_rnn_embed=None, is_train=True, reuse=False):
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    s = image_size # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if net_rnn_embed is not None:
            net_rnn_embed = DenseLayer(net_rnn_embed, n_units=t_dim,
                act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_rnn_embed], concat_dim=1, name='g_concat_z_seq')
        else:
            print("No text info is used, i.e. DCGAN")

        net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
                W_init=w_init, b_init=None, name='g_h0/dense')
        net_h0 = BatchNormLayer(net_h0,  #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')

        net_h1 = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1/batch_norm')
        net_h2 = Conv2d(net_h1, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h2/batch_norm')
        net_h3 = Conv2d(net_h2, gf_dim*8, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, # act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h3/batch_norm')
        net_h3 = ElementwiseLayer(layer=[net_h3, net_h0], combine_fn=tf.add, name='g_h3/add')
        net_h3.outputs = tf.nn.relu(net_h3.outputs)

        #
        net_h4 = UpSampling2dLayer(net_h3, size=[s8, s8], is_scale=False, method=1,
                align_corners=False, name='g_h4/upsample2d')
        net_h4 = Conv2d(net_h4, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4,# act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        net_h5 = Conv2d(net_h4, gf_dim, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h5/batch_norm')
        net_h6 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h6/conv2d')
        net_h6 = BatchNormLayer(net_h6, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h6/batch_norm')
        net_h7 = Conv2d(net_h6, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h7/conv2d')
        net_h7 = BatchNormLayer(net_h7, #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h7/batch_norm')
        net_h7 = ElementwiseLayer(layer=[net_h7, net_h4], combine_fn=tf.add, name='g_h7/add')
        net_h7.outputs = tf.nn.relu(net_h7.outputs)

        # net_h8 = DeConv2d(net_h7, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h8/decon2d')
        net_h8 = UpSampling2dLayer(net_h7, size=[s4, s4], is_scale=False, method=1,
                align_corners=False, name='g_h8/upsample2d')
        net_h8 = Conv2d(net_h8, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h8/conv2d')
        net_h8 = BatchNormLayer(net_h8, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h8/batch_norm')

        # net_h9 = DeConv2d(net_h8, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h9/decon2d')
        net_h9 = UpSampling2dLayer(net_h8, size=[s2, s2], is_scale=False, method=1,
                align_corners=False, name='g_h9/upsample2d')
        net_h9 = Conv2d(net_h9, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h9/conv2d')
        net_h9 = BatchNormLayer(net_h9, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h9/batch_norm')

        # net_ho = DeConv2d(net_h9, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        net_ho = UpSampling2dLayer(net_h9, size=[s, s], is_scale=False, method=1,
                align_corners=False, name='g_ho/upsample2d')
        net_ho = Conv2d(net_ho, c_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho, logits

def discriminator_txt2img_resnet(input_images, net_rnn_embed=None, is_train=True, reuse=False):
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

        net_h = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d2')
        net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm2')
        net_h = Conv2d(net_h, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d3')
        net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm3')
        net_h = Conv2d(net_h, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d4')
        net_h = BatchNormLayer(net_h, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm4')
        net_h3 = ElementwiseLayer(layer=[net_h3, net_h], combine_fn=tf.add, name='d_h3/add')
        net_h3.outputs = tl.act.lrelu(net_h3.outputs, 0.2)

        if net_rnn_embed is not None:
            net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, name='d_reduce_txt/dense')
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1)
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1)
            net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, s16, s16, 1], name='d_tiled_embeddings')
            net_h3_concat = ConcatLayer([net_h3, net_reduced_text], concat_dim=3, name='d_h3_concat')
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),
                    padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')
            net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')
        else:
            print("No text info is used, i.e. DCGAN")
        net_h4 = Conv2d(net_h3, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='d_h4/conv2d_2')
        # 1 x 1 x 1
        # net_h4 = FlattenLayer(net_h4, name='d_h4/flatten')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits

def cnn_encoder_resnet(input_images, is_train=True, reuse=False, name='cnn'):
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py  d_encode_image
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        # (nc) x 64 x 64
        net_in = InputLayer(input_images, name='p_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='p_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='p_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='p_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='p_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm')

        net_h = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='p_h3/conv2d2')
        net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm2')
        net_h = Conv2d(net_h, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='p_h3/conv2d3')
        net_h = BatchNormLayer(net_h, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm3')
        net_h = Conv2d(net_h, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='p_h3/conv2d4')
        net_h = BatchNormLayer(net_h, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p_h3/batchnorm4')
        net_h3 = ElementwiseLayer(layer=[net_h3, net_h], combine_fn=tf.add, name='p_h3/add')
        net_h3.outputs = tl.act.lrelu(net_h3.outputs, 0.2)

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

## stack GAN ===================================================================
def stackG_256(inputs, net_rnn, is_train, reuse):
    """ 64x64-->256x256 """
    # line 185 https://github.com/hanzhanggit/StackGAN/blob/master/stageII/model.py
    #           https://github.com/hanzhanggit/StackGAN/blob/master/misc/custom_ops.py
    gf_dim = 128
    # df_dim = 128
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("stackG", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(inputs, name='stackG_input/images')
        # net_h0 = Conv2d(net_in, df_dim, (52, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
        #                 padding='SAME', W_init=w_init, name='stackG_h0/conv2d')
        ## downsampling
        net_h0 = Conv2d(net_in, gf_dim, (3, 3), (1, 1), act=tf.nn.relu,
                padding='SAME', W_init=w_init, name='stackG_p0/conv2d')

        net_h1 = Conv2d(net_h0, gf_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackG_p1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='stackG_p1/batchnorm')
        net_h2 = Conv2d(net_h1, gf_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackG_p2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='stackG_p2/batchnorm')

        # print(net_h2.outputs)
    # def hr_g_encode_image(self, x_var):
    #     output_tensor = \
    #         (pt.wrap(x_var).  # -->s * s * 3
    #          custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).  # s * s * gf_dim
    #          apply(tf.nn.relu).
    #          custom_conv2d(self.gf_dim * 2, k_h=4, k_w=4).  # s2 * s2 * gf_dim * 2
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          custom_conv2d(self.gf_dim * 4, k_h=4, k_w=4).  # s4 * s4 * gf_dim * 4
    #          conv_batch_norm().
    #          apply(tf.nn.relu))
    #     return output_tensor

        # exit()
        ## join image and text
        if net_rnn is not None:
            net_rnn = InputLayer(net_rnn.outputs, name='stackG_join/input_text')
            net_reduced_text = DenseLayer(net_rnn, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, b_init=None, name='stackG_join_reduce_txt/dense')
            # print('t1',net_reduced_text.outputs)
            # net_reduced_text = net_rnn_embed  # if reduce_txt in rnn_embed
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1)  # you can use ExpandDimsLayer and TileLayer instead
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 2)
            net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 16, 16, 1], name='stackG_join_tile')
            # print('t2',net_reduced_text.outputs)
            net_h3_concat = ConcatLayer([net_h2, net_reduced_text], concat_dim=3, name='stackG_h3_concat') # (64, 4, 4, 640)
            # print('con1',net_h3_concat.outputs)
            # net_h3_concat = net_h3 # no text info
            net_h3 = Conv2d(net_h3_concat, gf_dim*4, (3, 3), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='stackG_join/conv2d')
            # print('con2',net_h3.outputs)
            net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu,
                   is_train=is_train, gamma_init=gamma_init, name='stackG_join/batch_norm')
        else:
            print("No text info will be used, i.e. normal DCGAN")
    # def hr_g_joint_img_text(self, x_c_code):
    #     output_tensor = \
    #         (pt.wrap(x_c_code).  # -->s4 * s4 * (ef_dim+gf_dim*4)
    #          custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).  # s4 * s4 * gf_dim * 4
    #          conv_batch_norm().
    #          apply(tf.nn.relu))
    #     return output_tensor
        # print(net_h3.outputs)

        ## residual block x 4(for 64--256)
        for i in range(4):
            net_h = Conv2d(net_h3, gf_dim*4, (3, 3), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='stackG_residual{}/conv2d_1'.format(i))
            net_h = BatchNormLayer(net_h, act=tf.nn.relu,
                   is_train=is_train, gamma_init=gamma_init, name='stackG_residual{}/batch_norm_1'.format(i))
            net_h = Conv2d(net_h, gf_dim*4, (3, 3), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='stackG_residual{}/conv2d_2'.format(i))
            net_h = BatchNormLayer(net_h, #act=tf.nn.relu,
                   is_train=is_train, gamma_init=gamma_init, name='stackG_residual{}/batch_norm_2'.format(i))
            net_h3 = ElementwiseLayer(layer=[net_h3, net_h], combine_fn=tf.add, name='stackG_residual{}/add'.format(i))
            net_h3.outputs = tf.nn.relu(net_h3.outputs)
    # def residual_block(self, x_c_code):
    #     node0_0 = pt.wrap(x_c_code)  # -->s4 * s4 * gf_dim * 4
    #     node0_1 = \
    #         (pt.wrap(x_c_code).  # -->s4 * s4 * gf_dim * 4
    #          custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm())
    #     output_tensor = \
    #         (node0_0.
    #          apply(tf.add, node0_1).
    #          apply(tf.nn.relu))
    #     return output_tensor
        # print(net_h3.outputs)

        ## upsampling 16x16-->64x64
        # net_h4 = DeConv2d(net_h3, gf_dim*2, (4, 4), out_size=(32, 32), strides=(2, 2),    # 16x16--32x32
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='stackG_up/decon2d_1')
        # you can use UpSampling2dLayer tf.image.resize_nearest_neighbor (method=1) instead of DeConv2d
        net_h4 = UpSampling2dLayer(net_h3, size=[32, 32], is_scale=False, method=1, align_corners=False, name='stackG_up/upsample2d_1')
        net_h4 = Conv2d(net_h4, gf_dim*2, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='stackG_up/conv2d_1')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='stackG_up/batch_norm_1')

        # net_h4 = DeConv2d(net_h4, gf_dim, (4, 4), out_size=(64, 64), strides=(2, 2),    # 32x32--64x64
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='stackG_up/decon2d_2')
        net_h4 = UpSampling2dLayer(net_h4, size=[64, 64], is_scale=False, method=1, align_corners=False, name='stackG_up/upsample2d_2')
        net_h4 = Conv2d(net_h4, gf_dim, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='stackG_up/conv2d_2')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='stackG_up/batch_norm_2')

        ###
        # net_h4 = DeConv2d(net_h4, gf_dim//2, (4, 4), out_size=(128, 128), strides=(2, 2),    # 64x64--128x128
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='stackG_up/decon2d_3')
        net_h4 = UpSampling2dLayer(net_h4, size=[128, 128], is_scale=False, method=1, align_corners=False, name='stackG_up/upsample2d_3')
        net_h4 = Conv2d(net_h4, gf_dim//2, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='stackG_up/conv2d_3')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='stackG_up/batch_norm_3')

        net_h4 = DeConv2d(net_h4, gf_dim//4, (4, 4), out_size=(256, 256), strides=(2, 2),    # 128x128--256x256
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='stackG_up/decon2d_4')
            # net_h4 = UpSampling2dLayer(net_h4, size=[256, 256], is_scale=False, method=1, align_corners=False, name='stackG_up/upsample2d_4')
        net_h4 = Conv2d(net_h4, gf_dim//4, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='stackG_up/conv2d_4')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='stackG_up/batch_norm_4')

        # print(net_h4.outputs)

        ## down to 3 channels
        network = Conv2d(net_h4, 3, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='stackG_out/conv2d')

        # print(network.outputs)
        # exit()
    # def hr_generator(self, x_c_code):  # their code is for 64x64-->256x256
    #     output_tensor = \
    #         (pt.wrap(x_c_code).  # -->s4 * s4 * gf_dim*4
    #          # custom_deconv2d([0, self.s2, self.s2, self.gf_dim * 2], k_h=4, k_w=4).  # -->s2 * s2 * gf_dim*2
    #          apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
    #          custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          # custom_deconv2d([0, self.s, self.s, self.gf_dim], k_h=4, k_w=4).  # -->s * s * gf_dim
    #          apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
    #          custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          # custom_deconv2d([0, self.s * 2, self.s * 2, self.gf_dim // 2], k_h=4, k_w=4).  # -->2s * 2s * gf_dim/2
    #          apply(tf.image.resize_nearest_neighbor, [self.s * 2, self.s * 2]).
    #          custom_conv2d(self.gf_dim // 2, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          # custom_deconv2d([0, self.s * 4, self.s * 4, self.gf_dim // 4], k_h=4, k_w=4).  # -->4s * 4s * gf_dim//4
    #          apply(tf.image.resize_nearest_neighbor, [self.s * 4, self.s * 4]).
    #          custom_conv2d(self.gf_dim // 4, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).  # -->4s * 4s * 3
    #          apply(tf.nn.tanh))
    #     return output_tensor

    # class custom_deconv2d(pt.VarStoreMethod):
    #     def __call__(self, input_layer, output_shape,
    #                  k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
    #                  name="deconv2d"):
    #         output_shape[0] = input_layer.shape[0]
    #         ts_output_shape = tf.pack(output_shape)
    #         with tf.variable_scope(name):
    #             # filter : [height, width, output_channels, in_channels]
    #             w = self.variable('w', [k_h, k_w, output_shape[-1], input_layer.shape[-1]],
    #                               init=tf.random_normal_initializer(stddev=stddev))
    #
    #            deconv = tf.nn.conv2d_transpose(input_layer, w,
    #                                            output_shape=ts_output_shape,
    #                                            strides=[1, d_h, d_w, 1])
    #
    #             # biases = self.variable('biases', [output_shape[-1]], init=tf.constant_initializer(0.0))
    #             # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), [-1] + output_shape[1:])
    #             deconv = tf.reshape(deconv, [-1] + output_shape[1:])
    #
    #             return deconv
        logits = network.outputs
        network.outputs = tf.nn.tanh(network.outputs)                             # 188
        # exit(network.outputs)
    return network, logits

# def stackD_64(input_images, net_rnn_embed=None, is_train=True, reuse=False): # same as discriminator_txt2img
#     # IMPLEMENTATION based on : https://github.com/paarthneekhara/text-to-image/blob/master/model.py
#     #       https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     b_init = None # tf.constant_initializer(value=0.0)
#     gamma_init=tf.random_normal_initializer(1., 0.02)
#
#     with tf.variable_scope("stackD", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#
#         net_in = InputLayer(input_images, name='stackD_input/images')
#         net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
#                 padding='SAME', W_init=w_init, name='stackD_h0/conv2d')  # (64, 32, 32, 64)
#
#         net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h1/conv2d')
#         net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='stackD_h1/batchnorm') # (64, 16, 16, 128)
#
#         net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h2/conv2d')
#         net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='stackD_h2/batchnorm')    # (64, 8, 8, 256)
#
#         net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h3/conv2d')
#         net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init, name='stackD_h3/batchnorm') # (64, 4, 4, 512)  paper 4.1: when the spatial dim of the D is 4x4, we replicate the description embedding spatially and perform a depth concatenation
#
#         if net_rnn_embed is not None:
#             # paper : reduce the dim of description embedding in (seperate) FC layer followed by rectification
#             net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,
#                    act=lambda x: tl.act.lrelu(x, 0.2),
#                    W_init=w_init, b_init=None, name='stackD_reduce_txt/dense')
#             # net_reduced_text = net_rnn_embed  # if reduce_txt in rnn_embed
#             net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1)  # you can use ExpandDimsLayer and TileLayer instead
#             net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 2)
#             net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 4, 4, 1], name='stackD_tiled_embeddings')
#
#             net_h3_concat = ConcatLayer([net_h3, net_reduced_text], concat_dim=3, name='stackD_h3_concat') # (64, 4, 4, 640)
#             # net_h3_concat = net_h3 # no text info
#             net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1),
#                    padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h3/conv2d_2')   # paper 4.1: perform 1x1 conv followed by rectification and a 4x4 conv to compute the final score from D
#             net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
#                    is_train=is_train, gamma_init=gamma_init, name='stackD_h3/batch_norm_2') # (64, 4, 4, 512)
#         else:
#             print("No text info will be used, i.e. normal DCGAN")
#
#         net_h4 = FlattenLayer(net_h3, name='stackD_h4/flatten')          # (64, 8192)
#         net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
#                 W_init = w_init, name='stackD_h4/dense')
#         logits = net_h4.outputs
#         net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)  # (64, 1)
#     return net_h4, logits

def stackD_256(input_images, net_rnn_embed=None, is_train=True, reuse=False): # same as discriminator_txt2img
    """ 256x256 -> real fake """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("stackD", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_images, name='stackD_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='stackD_h0/conv2d')  # (64, 32, 32, 64)

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h1/batchnorm') # (64, 16, 16, 128)

        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h2/batchnorm')    # (64, 8, 8, 256)

        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h3/batchnorm') # (64, 4, 4, 512)  paper 4.1: when the spatial dim of the D is 4x4, we replicate the description embedding spatially and perform a depth concatenation

        net_h4 = Conv2d(net_h3, df_dim*16, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h4/batchnorm')

        net_h5 = Conv2d(net_h4, df_dim*32, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h5/batchnorm')

        net_h6 = Conv2d(net_h5, df_dim*16, (1, 1), (1, 1), act=None,                        # filter (1, 1) for XXX
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h6/conv2d')
        net_h6 = BatchNormLayer(net_h6, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h6/batchnorm')

        net_h7 = Conv2d(net_h6, df_dim*8, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h7/conv2d')
        net_h7 = BatchNormLayer(net_h7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h7/batchnorm')
    # line 270  https://github.com/hanzhanggit/StackGAN/blob/master/stageII/model.py
        # node1_0 = \
        #     (pt.template("input").  # 4s * 4s * 3
        #      custom_conv2d(self.df_dim, k_h=4, k_w=4).  # 2s * 2s * df_dim
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).  # s * s * df_dim*2
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).  # s2 * s2 * df_dim*4
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).  # s4 * s4 * df_dim*8
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 16, k_h=4, k_w=4).  # s8 * s8 * df_dim*16
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 32, k_h=4, k_w=4).  # s16 * s16 * df_dim*32
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 16, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * df_dim*16
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * df_dim*8
        #      conv_batch_norm())

        net_h8 = Conv2d(net_h7, df_dim*2, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h8/conv2d')
        net_h8 = BatchNormLayer(net_h8, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h8/batchnorm')

        net_h9 = Conv2d(net_h8, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h9/conv2d')
        net_h9 = BatchNormLayer(net_h9, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h9/batchnorm')

        net_h10 = Conv2d(net_h9, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h10/conv2d')
        net_h10 = BatchNormLayer(net_h10, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h10/batchnorm')
        # node1_1 = \
        #     (node1_0.
        #      custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
        #      conv_batch_norm())

        net_h11 = ElementwiseLayer(layer=[net_h7, net_h10], combine_fn=tf.add, name='stackD_residual/add')
        net_h11.outputs = tl.act.lrelu(net_h11.outputs, 0.2)
        # node1 = \
        #     (node1_0.
        #      apply(tf.add, node1_1).
        #      apply(leaky_rectify, leakiness=0.2))

        # print(net_h11.outputs)
        # net_h11 = ExpandDimsLayer(net_h11, 1, name='stackD_expand1')
        # print(net_h11.outputs)
        # net_h11 = ExpandDimsLayer(net_h11, 1, name='stackD_expand2')
        # net_h11 = TileLayer(net_h11, [1, 4, 4, 1], name='stackD_tile')

        # exit(net_h11.outputs)

        if net_rnn_embed is not None:
            # paper : reduce the dim of description embedding in (seperate) FC layer followed by rectification
            net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, b_init=None, name='stackD_reduce_txt/dense')
            # net_reduced_text = net_rnn_embed  # if reduce_txt in rnn_embed
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1)  # you can use ExpandDimsLayer and TileLayer instead
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 2)
            net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 4, 4, 1], name='stackD_tiled_embeddings')

            net_h11_concat = ConcatLayer([net_h11, net_reduced_text], concat_dim=3, name='stackD_h11_concat') # (64, 4, 4, 640)
            # net_h3_concat = net_h3 # no text info
            net_h11 = Conv2d(net_h11_concat, df_dim*8, (1, 1), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h11/conv2d_2')   # paper 4.1: perform 1x1 conv followed by rectification and a 4x4 conv to compute the final score from D
            net_h11 = BatchNormLayer(net_h11, act=lambda x: tl.act.lrelu(x, 0.2),
                   is_train=is_train, gamma_init=gamma_init, name='stackD_h11/batch_norm_2') # (64, 4, 4, 512)
        else:
            print("No text info will be used, i.e. normal DCGAN")
        # c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        # c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])  # s16 * s16 * ef_dim
        #
        # x_c_code = tf.concat(3, [x_code, c_code])

        net_h12 = FlattenLayer(net_h11, name='stackD_h4/flatten')          # (64, 8192)
        net_h12 = DenseLayer(net_h12, n_units=1, act=tf.identity,
                W_init = w_init, name='stackD_h4/dense')
        logits = net_h12.outputs
        net_h12.outputs = tf.nn.sigmoid(net_h12.outputs)  # (64, 1)

    return net_h12, logits

def cnn_encoder_256(input_images, net_rnn_embed=None, is_train=True, reuse=False, name='image_encoder'):
    """ 256x256 --> z noise , modify from stackD_256 """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_images, name='cnn_encoder_256_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='cnn_encoder_256_h0/conv2d')  # (64, 32, 32, 64)

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='cnn_encoder_256_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnn_encoder_256_h1/batchnorm') # (64, 16, 16, 128)

        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='cnn_encoder_256_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h2/batchnorm')    # (64, 8, 8, 256)

        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h3/batchnorm') # (64, 4, 4, 512)  paper 4.1: when the spatial dim of the D is 4x4, we replicate the description embedding spatially and perform a depth concatenation

        net_h4 = Conv2d(net_h3, df_dim*16, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h4/batchnorm')

        net_h5 = Conv2d(net_h4, df_dim*32, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h5/batchnorm')

        net_h6 = Conv2d(net_h5, df_dim*16, (1, 1), (1, 1), act=None,                        # filter (1, 1) for XXX
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h6/conv2d')
        net_h6 = BatchNormLayer(net_h6, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h6/batchnorm')

        net_h7 = Conv2d(net_h6, df_dim*8, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h7/conv2d')
        net_h7 = BatchNormLayer(net_h7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='stackD_h7/batchnorm')
    # line 270  https://github.com/hanzhanggit/StackGAN/blob/master/stageII/model.py
        # node1_0 = \
        #     (pt.template("input").  # 4s * 4s * 3
        #      custom_conv2d(self.df_dim, k_h=4, k_w=4).  # 2s * 2s * df_dim
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).  # s * s * df_dim*2
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).  # s2 * s2 * df_dim*4
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).  # s4 * s4 * df_dim*8
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 16, k_h=4, k_w=4).  # s8 * s8 * df_dim*16
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 32, k_h=4, k_w=4).  # s16 * s16 * df_dim*32
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 16, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * df_dim*16
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * df_dim*8
        #      conv_batch_norm())

        net_h8 = Conv2d(net_h7, df_dim*2, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='cnn_encoder_256_h8/conv2d')
        net_h8 = BatchNormLayer(net_h8, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnn_encoder_256_h8/batchnorm')

        net_h9 = Conv2d(net_h8, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='cnn_encoder_256_h9/conv2d')
        net_h9 = BatchNormLayer(net_h9, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnn_encoder_256_h9/batchnorm')

        net_h10 = Conv2d(net_h9, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='cnn_encoder_256_h10/conv2d')
        net_h10 = BatchNormLayer(net_h10, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnn_encoder_256_h10/batchnorm')
        # node1_1 = \
        #     (node1_0.
        #      custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
        #      conv_batch_norm().
        #      apply(leaky_rectify, leakiness=0.2).
        #      custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
        #      conv_batch_norm())

        net_h11 = ElementwiseLayer(layer=[net_h7, net_h10], combine_fn=tf.add, name='cnn_encoder_256_residual/add')
        net_h11.outputs = tl.act.lrelu(net_h11.outputs, 0.2)
        # node1 = \
        #     (node1_0.
        #      apply(tf.add, node1_1).
        #      apply(leaky_rectify, leakiness=0.2))

        ## DH add for deeper network =======================================
        net_h11 = Conv2d(net_h11, df_dim*4, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='cnn_encoder_256_h11/conv2d')
        net_h11 = BatchNormLayer(net_h11, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnn_encoder_256_h11/batchnorm')
        for i in range(4): # DH add for deeper network
            net_h = Conv2d(net_h11, df_dim*4, (3, 3), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='cnn_encoder_256_residual{}/conv2d_1'.format(i))
            net_h = BatchNormLayer(net_h, act=tf.nn.relu,
                   is_train=is_train, gamma_init=gamma_init, name='cnn_encoder_256_residual{}/batch_norm_1'.format(i))
            net_h = Conv2d(net_h, df_dim*4, (3, 3), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='cnn_encoder_256_residual{}/conv2d_2'.format(i))
            net_h = BatchNormLayer(net_h, #act=tf.nn.relu,
                   is_train=is_train, gamma_init=gamma_init, name='cnn_encoder_256_residual{}/batch_norm_2'.format(i))
            net_h11 = ElementwiseLayer(layer=[net_h11, net_h], combine_fn=tf.add, name='cnn_encoder_256_residual{}/add'.format(i))
            net_h11.outputs = tf.nn.relu(net_h11.outputs)
        ## end of DH add

        # print(net_h11.outputs)
        # net_h11 = ExpandDimsLayer(net_h11, 1, name='stackD_expand1')
        # print(net_h11.outputs)
        # net_h11 = ExpandDimsLayer(net_h11, 1, name='stackD_expand2')
        # net_h11 = TileLayer(net_h11, [1, 4, 4, 1], name='stackD_tile')

        # exit(net_h11.outputs)

        # if net_rnn_embed is not None:
        #     # paper : reduce the dim of description embedding in (seperate) FC layer followed by rectification
        #     net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,
        #            act=lambda x: tl.act.lrelu(x, 0.2),
        #            W_init=w_init, b_init=None, name='stackD_reduce_txt/dense')
        #     # net_reduced_text = net_rnn_embed  # if reduce_txt in rnn_embed
        #     net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1)  # you can use ExpandDimsLayer and TileLayer instead
        #     net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 2)
        #     net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 4, 4, 1], name='stackD_tiled_embeddings')
        #
        #     net_h11_concat = ConcatLayer([net_h11, net_reduced_text], concat_dim=3, name='stackD_h11_concat') # (64, 4, 4, 640)
        #     # net_h3_concat = net_h3 # no text info
        #     net_h11 = Conv2d(net_h11_concat, df_dim*8, (1, 1), (1, 1),
        #            padding='SAME', W_init=w_init, b_init=b_init, name='stackD_h11/conv2d_2')   # paper 4.1: perform 1x1 conv followed by rectification and a 4x4 conv to compute the final score from D
        #     net_h11 = BatchNormLayer(net_h11, act=lambda x: tl.act.lrelu(x, 0.2),
        #            is_train=is_train, gamma_init=gamma_init, name='stackD_h11/batch_norm_2') # (64, 4, 4, 512)
        # else:
        #     print("No text info will be used, i.e. normal DCGAN")
        # c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        # c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])  # s16 * s16 * ef_dim
        #
        # x_c_code = tf.concat(3, [x_code, c_code])

        net_h12 = FlattenLayer(net_h11, name='cnn_encoder_256_h4/flatten')          # (64, 8192)
        net_h12 = DenseLayer(net_h12, n_units=z_dim, act=tf.identity,
                W_init = w_init, name='cnn_encoder_256_h4/dense')
        # logits = net_h12.outputs
        # net_h12.outputs = tf.nn.sigmoid(net_h12.outputs)  # (64, 1)

    return net_h12#, logits


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
