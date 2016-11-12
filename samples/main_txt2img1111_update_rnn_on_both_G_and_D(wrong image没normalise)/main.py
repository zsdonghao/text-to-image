#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
import numpy as np
import scipy
from scipy.io import loadmat
import time
import os
import re
import nltk
import random

from utils import *


""" Generative Adversarial Text to Image Synthesis

Downlaod Oxford 102 flowers dataset and caption
-------------------------------------------------
Flowers  : http://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/
        paste it in 102flowers/102flowers/*jpg
Captions : https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view
        paste it in 102flowers/text_c10/class_*

Code References
---------------
- GAN-CLS by TensorFlow
- https://github.com/paarthneekhara/text-to-image/blob/master/train.py
- https://github.com/paarthneekhara/text-to-image/blob/master/model.py
- https://github.com/paarthneekhara/text-to-image/blob/master/Utils/ops.py
"""
###======================== PREPARE DATA ====================================###
## Directory of Oxford 102 flowers dataset
if True:
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '102flowers/102flowers')
    caption_dir = os.path.join(cwd, '102flowers/text_c10')
    VOC_FIR = cwd + '/vocab.txt'

    ## load captions
    caption_sub_dir = load_folder_list( caption_dir )
    captions_dict = {}
    processed_capts = []
    for sub_dir in caption_sub_dir: # get caption file list
        with tl.ops.suppress_stdout():
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
            for i, f in enumerate(files):
                file_dir = os.path.join(sub_dir, f)
                key = int(re.findall('\d+', f)[0])
                t = open(file_dir,'r')
                lines = []
                for line in t:
                    lines.append(line.rstrip()) # remove \n
                    processed_capts.append(tl.nlp.process_sentence(line.rstrip(), start_word="<S>", end_word="</S>"))
                assert len(lines) == 10, "Every flower image have 10 captions"
                captions_dict[key] = lines
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))


    ## build vocab
    _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
    vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")

    ## store all captions ids in list
    captions_ids = []
    for key, value in captions_dict.iteritems():
        for v in value:
            captions_ids.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] )
            # print(v)              # prominent purple stigma,petals are white inc olor
            # print(captions_ids)   # [[152, 19, 33, 15, 3, 8, 14, 719, 723]]
            # exit()
    captions_ids = np.asarray(captions_ids)
    print(" * tokenized %d captions" % len(captions_ids))

    ## check
    img_capt = captions_dict[1][1]
    print("img_capt: %s" % img_capt)
    print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
    img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
    print("img_capt_ids: %s" % img_capt_ids)
    print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])

    ## load images
    with tl.ops.suppress_stdout():  # get image files list
        imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))
    print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
    s = time.time()
    images = []
    for name in imgs_title_list:
        img = scipy.misc.imread( os.path.join(img_dir, name) )
        img = tl.prepro.imresize(img, size=[64, 64])    # (64, 64, 3)
        img = img.astype(np.float32)
        images.append(img)
    images = np.asarray(images)
    print(" * loading and resizing took %ss" % (time.time()-s))

    n_images = len(captions_dict)
    n_captions = len(captions_ids)
    n_captions_per_image = len(lines) # 10
# ## check the first example
# tl.visualize.frame(I=images[0], second=5, saveable=True, name='temp', cmap=None)
# for cap in captions_dict[1]:
#     print(cap)
# print(captions_ids[0:10])
# for ids in captions_ids[0:10]:
#     print([vocab.id_to_word(id) for id in ids])
# print_dict(captions_dict)

###======================== DEFIINE MODEL ===================================###
batch_size = 64
vocab_size = 8000
word_embedding_size = 256    # Hao
z_dim = 100         # Noise dimension
t_dim = 256 /2        # Text feature dimension # paper said 128
image_size = 64     # 64 x 64
c_dim = 3           # for rgb
gf_dim = 64         # Number of conv in the first layer generator 64
df_dim = 64         # Number of conv in the first layer discriminator 64
# gfc_dim = 1024      # Dimension of gen untis for for fully connected layer 1024
# caption_vector_length = 2400 # Caption Vector Length 2400   Hao : I use word-based dynamic_rnn

print("n_captions: %d batch_size: %d n_captions_per_image: %d" % (n_captions, batch_size, n_captions_per_image))

# ## generate a random batch
# idexs = generate_random_int(0, n_captions, batch_size)
# idexs = [i for i in range(0,100)]
# print(idexs)
# b_seqs = captions_ids[idexs]
# b_images = images[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
# print("before padding %s" % b_seqs)
# b_seqs = tl.prepro.pad_sequences(b_seqs, padding='post')
# print("after padding %s" % b_seqs)
# # print(input_images.shape)   # (64, 64, 64, 3)
# for ids in b_seqs:
#     print([vocab.id_to_word(id) for id in ids])
# print(np.max(b_images), np.min(b_images), b_images.shape)
# tl.visualize.images2d(b_images, second=5, saveable=True, name='temp2')
# exit()



def rnn_embed(input_seqs, is_train, reuse):
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
                     cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
                     n_hidden = word_embedding_size,
                     dropout = (0.7 if is_train else None),
                     initializer = w_init,
                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
                     return_last = True,
                     name = 'dynamicrnn',)
        # G and D share the same dense layer for reduce_txt, but # paper : reduce the dim of description embedding in (seperate) FC layer followed by rectification
        # network = DenseLayer(network, n_units=t_dim,
        #         act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='reduce_txt/dense')
    return network

def generator_txt2img(input_z, net_rnn_embed=None, is_train=True, reuse=False):
    """IMPLEMENTATION based on : https://github.com/paarthneekhara/text-to-image/blob/master/model.py
    """
    s = image_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_input_z = InputLayer(input_z, name='g_inputz')

        if net_rnn_embed is not None:
            # paper : reduce the dim of description embedding in (seperate) FC layer followed by rectification
            net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim, act= lambda x: tl.act.lrelu(x, 0.2),  # local reduce_txt, remove if reduce_txt in rnn_embed
                    W_init = w_init, name='g_reduce_text/dense')                                              # local reduce_txt, remove if reduce_txt in rnn_embed
            # net_reduced_text = net_rnn_embed  # if reduce_txt in rnn_embed
            net_z_concat = ConcatLayer([net_input_z, net_reduced_text], concat_dim=1, name='g_concat_z_seq')
        else:
            print("No text info will be used, i.e. normal DCGAN")
            net_z_concat = net_input_z

        net_h0 = DenseLayer(net_z_concat, gf_dim*8*s16*s16,
                act = tf.identity, W_init = w_init, name='g_h0/dense')                  # (64, 8192)
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                padding = 'SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                padding = 'SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
                padding = 'SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size = (s, s), strides = (2, 2),
                padding = 'SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_h4/decon2d')
        logits = net_h4.outputs
        # net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)  # DCGAN uses tanh
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits

def discriminator_txt2img(input_images, net_rnn_embed=None, is_train=True, reuse=False):
    """IMPLEMENTATION based on : https://github.com/paarthneekhara/text-to-image/blob/master/model.py
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in_img = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in_img, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')  # (64, 32, 32, 64)

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm') # (64, 16, 16, 128)

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')    # (64, 8, 8, 256)

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), padding='SAME', W_init=w_init, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm') # (64, 4, 4, 512)

        if net_rnn_embed is not None:
            # paper : reduce the dim of description embedding in (seperate) FC layer followed by rectification
            net_reduced_text = DenseLayer(net_rnn_embed, n_units=t_dim,                           # local reduce_txt, remove if reduce_txt in rnn_embed
                   act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='d_reduce_txt/dense') # local reduce_txt, remove if reduce_txt in rnn_embed
            # net_reduced_text = net_rnn_embed  # if reduce_txt in rnn_embed
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 1)
            net_reduced_text.outputs = tf.expand_dims(net_reduced_text.outputs, 2)
            net_reduced_text.outputs = tf.tile(net_reduced_text.outputs, [1, 4, 4, 1], name='d_tiled_embeddings')

            net_h3_concat = ConcatLayer([net_h3, net_reduced_text], concat_dim=3, name='d_h3_concat') # (64, 4, 4, 640)
            # net_h3_concat = net_h3 # no text info
            net_h3 = Conv2d(net_h3_concat, df_dim*8, (1, 1), (1, 1), padding='SAME', W_init=w_init, name='d_h3/conv2d_2')
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

# with tf.device("/gpu:0"):
##
# https://github.com/paarthneekhara/text-to-image/blob/master/train.py
# https://github.com/paarthneekhara/text-to-image/blob/master/model.py
# https://github.com/paarthneekhara/text-to-image/blob/master/Utils/ops.py
## build_model
t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3 ], name = 'wrong_image')    # remove if DCGAN only
t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')     # remove if DCGAN only
t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')


## training inference for training DCGAN
# from dcgan_model import *
# net_fake_image, _ = generator_dcgan(t_z, is_train=True, reuse=False)
# _, disc_fake_image_logits = discriminator_dcgan(net_fake_image.outputs, is_train=True, reuse=False)
# _, disc_real_image_logits = discriminator_dcgan(t_real_image, is_train=True, reuse=True)
## training inference for txt2img
net_read_caption = rnn_embed(t_real_caption, is_train=True, reuse=False)   # remove if DCGAN only
net_fake_image, _ = generator_txt2img(t_z,
                net_read_caption,                                       # remove if DCGAN only
                is_train=True, reuse=False)
_, disc_fake_image_logits  = discriminator_txt2img(net_fake_image.outputs,
                net_read_caption,                                       # remove if DCGAN only
                is_train=True, reuse=False)
_, disc_real_image_logits = discriminator_txt2img(t_real_image,
                net_read_caption,                                          # remove if DCGAN only
                is_train=True, reuse=True)
_, disc_wrong_image_logits = discriminator_txt2img(t_wrong_image,                 # remove if DCGAN only
                net_read_caption,                                            # remove if DCGAN only
                is_train=True, reuse=True)                               # remove if DCGAN only

## testing inference for DCGAN
# net_generator, _ = generator_dcgan(t_z, is_train=False, reuse=True)
## testing inference for txt2img
net_generator, _ = generator_txt2img(t_z,
                rnn_embed(t_real_caption, is_train=False, reuse=True), # remove if DCGAN only
                is_train=False, reuse=True)


## loss for DCGAN
# d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits, tf.ones_like(disc_real_image_logits)))    # real == 1
# d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits)))     # fake == 0
# d_loss = d_loss_real + d_loss_fake
# g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits)))

## loss for txt2img
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits))) # real == 1, fake == 0

d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits, tf.ones_like(disc_real_image_logits)))
d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image_logits)))
d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits)))

d_loss = d_loss1 + d_loss2 + d_loss3

net_fake_image.print_params(False)
net_fake_image.print_layers()
# exit()

####======================== DEFINE TRAIN OPTS ==========================###
## Cost   real == 1, fake == 0
lr = 0.0002
beta1 = 0.5
e_vars = tl.layers.get_variables_with_name('rnn', True, True)           #  remove if DCGAN only
d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
g_vars = tl.layers.get_variables_with_name('generator', True, True)

## When should we update word embedding and rnn ?
# update rnn in both D and G, ouput blurred flower but didn't match with txt yet
# update rnn only in G, output nothing but noise
# update rnn only in D, output visible image but don't match with text, low d_loss and high g_loss
## clip_grads for RNN ?
d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(d_loss, var_list=d_vars + e_vars)
g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(g_loss, var_list=g_vars + e_vars)

###============================ TRAINING ====================================###
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# sess=tf.Session()
# tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
# sess.run(tf.initialize_all_variables())

## seed for generation, z and sentence ids
sample_size = batch_size
sample_seed = np.random.uniform(low=-1, high=1, size=(sample_size, z_dim)).astype(np.float32)               # paper said [0, 1]
sample_sentence = ["this white and yellow flower have thin white petals and a round yellow stamen", \
                    "the flower has petals that are bright pinkish purple with white stigma"] * 32
for i, sentence in enumerate(sample_sentence):
    # sample_sentence[i] = tl.nlp.process_sentence(sentence, start_word=None, end_word=None)
    sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)]
    # print(sentence)
    # print(sample_sentence[i])
sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

n_epoch = 600   # 600 when pre-trained rnn
print_freq = 1
n_batch_epoch = int(n_images / batch_size)
for epoch in range(n_epoch):
    start_time = time.time()
    train_loss = 0
    for step in range(n_batch_epoch):
        step_time = time.time()
        ## get real image + matched text
        idexs = generate_random_int(min=0, max=n_captions-1, number=batch_size)
        b_real_caption = captions_ids[idexs]                                                                      # remove if DCGAN only
        b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')     # matched text  (64, any)    # remove if DCGAN only
        b_real_images = images[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]   # real images   (64, 64, 64, 3)
        ## get wrong caption
        # idexs = generate_random_int(min=0, max=n_captions-1, number=batch_size)
        # b_wrong_caption = captions_ids[idexs]
        # b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')                                    # mismatched text
        ## get wrong image
        idexs = generate_random_int(min=0, max=n_images-1, number=batch_size)        # remove if DCGAN only
        b_wrong_images = images[idexs]                                               # remove if DCGAN only
        ## get noise
        b_z = np.random.uniform(low=-1, high=1, size=[batch_size, z_dim]).astype(np.float32)       # paper said [0, 1]
        ## check data
        # print(np.min(b_real_images), np.max(b_real_images), b_real_images.shape)    # [0, 1] (64, 64, 64, 3)
        # for i, seq in enumerate(b_real_caption):
        #     print(seq)
        #     print(" ".join([vocab.id_to_word(id) for id in seq]))
        # exit()

        ## updates the discriminator
        # for _ in range(10):
        b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # random flip left and right    # https://github.com/paarthneekhara/text-to-image/blob/master/Utils/image_processing.py
        errD, _ = sess.run([d_loss, d_optim], feed_dict={
                        t_real_image : b_real_images,
                        t_wrong_image : b_wrong_images,     # remove if DCGAN only
                        t_real_caption : b_real_caption,    # remove if DCGAN only
                        t_z : b_z})
        # if epoch % 5 == 0:   # Hao : skip training G
            ## updates the generator
        for _ in range(2):
            errG, _ = sess.run([g_loss, g_optim], feed_dict={
                            # t_real_image : b_real_images,
                            # t_wrong_image : b_wrong_images,
                            t_real_caption : b_real_caption,    # remove if DCGAN only
                            t_z : b_z})

        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG))

        if np.isnan(errD) or np.isnan(errG):
            exit(" ** NaN error, stop training")

    if (epoch + 1) % print_freq == 0:
        print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
        img_gen = sess.run(net_generator.outputs, feed_dict={t_z: sample_seed,
                                                    t_real_caption: sample_sentence  # remove if DCGAN only
                                                    })
        # print(b_real_images[0])
        print('real:', b_real_images[0].shape, np.min(b_real_images[0]), np.max(b_real_images[0]))
        # print(img_gen[0])
        print('generate:', img_gen[0].shape, np.min(img_gen[0]), np.max(img_gen[0]))
        img_gen = threading_data(img_gen, prepro_img, mode='rescale')
        # tl.visualize.frame(img_gen[0], second=0, saveable=True, name='e_%d_%s' % (epoch, " ".join([vocab.id_to_word(id) for id in sample_sentence[0]])) )
        save_images(img_gen, [8, 8],
                    '{}/train_{:02d}.png'.format('samples', epoch))
        # for i, img in enumerate(img_gen):
        #     tl.visualize.frame(img, second=0, saveable=True, name='epoch_%d_sample_%d_%s' % (epoch, i, [vocab.id_to_word(id) for id in sample_sentence[i]]) )




























































#
