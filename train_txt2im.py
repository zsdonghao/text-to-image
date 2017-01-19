#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time
import os
import re
import nltk
import random

from utils import *
from model import *

os.system("mkdir samples")
os.system("mkdir checkpoint")

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
## Load Oxford 102 flowers dataset
from data_loader import *

###======================== DEFIINE MODEL ===================================###
## you may want to see how the data augmentation work
# save_images(images[:64], [8, 8], 'temp.png')
# pre_img = threading_data(images[:64], prepro_img, mode='debug')
# save_images(pre_img, [8, 8], 'temp2.png')
# # print(images[:64].shape, np.min(images[:64]), np.max(images[:64]))
# print(pre_img.shape, np.min(pre_img), np.max(pre_img))
# exit()

## build model
t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3 ], name = 'wrong_image')    # remove if DCGAN only
t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')     # remove if DCGAN only
t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

## training inference for text-to-image mapping   2017
net_cnn = cnn_encoder(t_real_image, is_train=True, reuse=False)
x = net_cnn.outputs
v = rnn_embed(t_real_caption, is_train=True, reuse=False).outputs
x_w = cnn_encoder(t_wrong_image, is_train=True, reuse=True).outputs
v_w = rnn_embed(t_wrong_caption, is_train=True, reuse=True).outputs

alpha = 0.2 # margin alpha
e_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
            tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x_w, v)))

## training inference for training DCGAN
# from dcgan_model import *
# net_fake_image, _ = generator_dcgan(t_z, is_train=True, reuse=False)
# _, disc_fake_image_logits = discriminator_dcgan(net_fake_image.outputs, is_train=True, reuse=False)
# _, disc_real_image_logits = discriminator_dcgan(t_real_image, is_train=True, reuse=True)
## training inference for txt2img
net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=True, return_embed=False)   # remove if DCGAN only
net_fake_image, _ = generator_txt2img(t_z,
                net_rnn,                                       # remove if DCGAN only
                is_train=True, reuse=False)
net_d, disc_fake_image_logits = discriminator_txt2img(
                net_fake_image.outputs,
                net_rnn,                                       # remove if DCGAN only
                is_train=True, reuse=False)
_, disc_real_image_logits = discriminator_txt2img(
                t_real_image,
                net_rnn,                                          # remove if DCGAN only
                is_train=True, reuse=True)
_, disc_wrong_image_logits = discriminator_txt2img( # CLS
                t_wrong_image,                                          # remove if DCGAN only
                net_rnn,                                            # remove if DCGAN only
                is_train=True, reuse=True)                               # remove if DCGAN only

## testing inference for DCGAN
# net_g, _ = generator_dcgan(t_z, is_train=False, reuse=True)
## testing inference for txt2img
net_g, _ = generator_txt2img(t_z,
                rnn_embed(t_real_caption, is_train=False, reuse=True, return_embed=False), # remove if DCGAN only
                is_train=False, reuse=True)

d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits, tf.ones_like(disc_real_image_logits)))
d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image_logits)))    # for CLS, if set it to zero, it is the same with normal DCGAN
d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits)))

d_loss = d_loss1 + d_loss2 + d_loss3

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits))) # real == 1, fake == 0

# net_fake_image.print_params(False)
# net_fake_image.print_layers()
# exit()

####======================== DEFINE TRAIN OPTS ==========================###
## Cost   real == 1, fake == 0
lr = 0.0002
beta1 = 0.5
n_g_batch = 2   # update G, x time per batch
c_vars = tl.layers.get_variables_with_name('cnn', True, True)
e_vars = tl.layers.get_variables_with_name('rnn', True, True)
d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
g_vars = tl.layers.get_variables_with_name('generator', True, True)

# grads = tf.gradients(d_loss, d_vars + e_vars)
# grads, _ = tf.clip_by_global_norm(tf.gradients(d_loss, d_vars + e_vars), 30)
# optimizer = tf.train.AdamOptimizer(1e-4, beta1=beta1)
# d_optim = optimizer.apply_gradients(zip(grads, d_vars + e_vars))
#
# grads = tf.gradients(g_loss, g_vars)
# grads, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars), 30)
# optimizer = tf.train.AdamOptimizer(1e-4, beta1=beta1)
# g_optim = optimizer.apply_gradients(zip(grads, g_vars))

d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(d_loss, var_list=d_vars )
g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(g_loss, var_list=g_vars )
e_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(e_loss, var_list=e_vars + c_vars)

###============================ TRAINING ====================================###
sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
tl.layers.initialize_global_variables(sess)

save_dir = "checkpoint"
if not os.path.exists(save_dir):
    print("[!] Folder (%s) is not exist, creating it ..." % save_dir)
    os.mkdir(save_dir)

# load the latest checkpoints
net_e_name = os.path.join(save_dir, 'net_e.npz')
net_c_name = os.path.join(save_dir, 'net_c.npz')
net_g_name = os.path.join(save_dir, 'net_g.npz')
net_d_name = os.path.join(save_dir, 'net_d.npz')
if False:
    if not (os.path.exists(net_e_name) and os.path.exists(net_c_name)):
        print("[!] Loading RNN and CNN checkpoints failed!")
    else:
        net_c_loaded_params = tl.files.load_npz(name=net_c_name)
        net_e_loaded_params = tl.files.load_npz(name=net_e_name)
        tl.files.assign_params(sess, net_c_loaded_params, net_cnn)
        tl.files.assign_params(sess, net_e_loaded_params, net_rnn)
        print("[*] Loading RNN and CNN checkpoints SUCCESS!")

    if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
        print("[!] Loading G and D checkpoints failed!")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_d_loaded_params = tl.files.load_npz(name=net_d_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        tl.files.assign_params(sess, net_d_loaded_params, net_d)
        print("[*] Loading G and D checkpoints SUCCESS!")

# sess=tf.Session()
# tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
# sess.run(tf.initialize_all_variables())

## seed for generation, z and sentence ids
sample_size = batch_size
sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
    # sample_seed = np.random.uniform(low=-1, high=1, size=(sample_size, z_dim)).astype(np.float32)        # paper said [0, 1]
# sample_sentence = ["this white and yellow flower have thin white petals and a round yellow stamen", \
#                     "the flower has petals that are bright pinkish purple with white stigma"] * 32
# sample_sentence = ["these flowers have petals that start off white in color and end in a dark purple towards the tips"] * 32 + \
#                     ["bright droopy yellow petals with burgundy streaks and a yellow stigma"] * 32
# sample_sentence = ["these white flowers have petals that start off white in color and end in a white towards the tips",
#                     "this yellow petals with burgundy streaks and a yellow stigma"] * 32
sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/8) + \
                  ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/8) + \
                  ["the petals on this flower are white with a yellow center"] * int(sample_size/8) + \
                  ["this flower has a lot of small round pink petals."] * int(sample_size/8) + \
                  ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/8) + \
                  ["the flower has yellow petals and the center of it is brown."] * int(sample_size/8) + \
                  ["this flower has petals that are blue and white."] * int(sample_size/8) +\
                  ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/8)

# sample_sentence = captions_ids_test[0:sample_size]
for i, sentence in enumerate(sample_sentence):
    print("seed: %s" % sentence)
    sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]    # add END_ID
    # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
    # print(sample_sentence[i])
sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')


n_epoch = 100   # 600 when pre-trained rnn
print_freq = 1
n_batch_epoch = int(n_images / batch_size)
for epoch in range(n_epoch):
    start_time = time.time()
    for step in range(n_batch_epoch):
        step_time = time.time()
        ## get matched text
        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
        b_real_caption = captions_ids_train[idexs]                                                                      # remove if DCGAN only
        b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')     # matched text  (64, any)    # remove if DCGAN only
        ## get real image
        b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]   # real images   (64, 64, 64, 3)
        ## get wrong caption
        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
        b_wrong_caption = captions_ids[idexs]
        b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')                                    # mismatched text
        ## get wrong image
        idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)        # remove if DCGAN only
        b_wrong_images = images_train[idexs2]                                               # remove if DCGAN only
        ## get noise
        b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
            # b_z = np.random.uniform(low=-1, high=1, size=[batch_size, z_dim]).astype(np.float32)       # paper said [0, 1], but [-1, 1] is better
        ## check data
        # print(np.min(b_real_images), np.max(b_real_images), b_real_images.shape)    # [0, 1] (64, 64, 64, 3)
        # for i, seq in enumerate(b_real_caption):
        #     # print(seq)
        #     print(i, " ".join([vocab.id_to_word(id) for id in seq]))
        # save_images(b_real_images, [8, 8], 'real_image.png')
        # exit()

        ## updates text-to-image mapping
        if epoch < 30:
            errE, _ = sess.run([e_loss, e_optim], feed_dict={
                                            t_real_image : b_real_images,
                                            t_wrong_image : b_wrong_images,
                                            t_real_caption : b_real_caption,
                                            t_wrong_caption : b_wrong_caption,
                                            # t_z : b_z # error
                                            })
            # total_e_loss += errE
        else:
            errE = 0

        ## updates D
        b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1]
        b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train')
        errD, _ = sess.run([d_loss, d_optim], feed_dict={
                        t_real_image : b_real_images,
                        t_wrong_image : b_wrong_images,     # remove if DCGAN only
                        t_real_caption : b_real_caption,    # remove if DCGAN only
                        t_z : b_z})
        ## updates G
        for _ in range(n_g_batch):
            errG, _ = sess.run([g_loss, g_optim], feed_dict={
                            t_real_caption : b_real_caption,    # remove if DCGAN only
                            t_z : b_z})

        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f, e_loss: %.8f" \
                    % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG, errE))

        # if np.isnan(errD) or np.isnan(errG):
        #     exit(" ** NaN error, stop training")

    if (epoch + 1) % print_freq == 0:
        print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
        img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs],
        # img_gen = sess.run(net_g.outputs,
                                    feed_dict={
                                    t_real_caption : sample_sentence,  # remove if DCGAN only
                                    t_z : sample_seed})

        # print(b_real_images[0])
        print('rnn:', np.min(rnn_out[0]), np.max(rnn_out[0]))   # -1.4121389, 1.4108921
        print('real:', b_real_images[0].shape, np.min(b_real_images[0]), np.max(b_real_images[0]))
        print('wrong:', b_wrong_images[0].shape, np.min(b_wrong_images[0]), np.max(b_wrong_images[0]))
        # print(img_gen[0])
        print('generate:', img_gen[0].shape, np.min(img_gen[0]), np.max(img_gen[0]))
        img_gen = threading_data(img_gen, prepro_img, mode='rescale')   # [-1, 1] --> [-1, 1]
        # tl.visualize.frame(img_gen[0], second=0, saveable=True, name='e_%d_%s' % (epoch, " ".join([vocab.id_to_word(id) for id in sample_sentence[0]])) )
        save_images(img_gen, [8, 8], '{}/train_{:02d}.png'.format('samples', epoch))
        # for i, img in enumerate(img_gen):
        #     tl.visualize.frame(img, second=0, saveable=True, name='epoch_%d_sample_%d_%s' % (epoch, i, [vocab.id_to_word(id) for id in sample_sentence[i]]) )
        # print(img_gen[:32])
        # print(img_gen[32:])
        # tl.visualize.images2d(images=img_gen, second=0.01, saveable=True, name='temp_generate', dtype=np.uint8)

        # b_real_images = threading_data(b_real_images, prepro_img, mode='rescale')
        # b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='rescale')
        # save_images(b_real_images, [8, 8], 'temp_real_image.png')
        # save_images(b_wrong_images, [8, 8], 'temp_wrong_image.png')

    if epoch % 5 == 0:
        tl.files.save_npz(net_cnn.all_params, name=net_c_name, sess=sess)
        tl.files.save_npz(net_rnn.all_params, name=net_e_name, sess=sess)
        tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
        tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
        print("[*] Saving checkpoints SUCCESS!")
        # net_c_name_e = os.path.join(save_dir, 'net_c_%d.npz' % epoch)
        # net_e_name_e = os.path.join(save_dir, 'net_e_%d.npz' % epoch)
        # net_g_name_e = os.path.join(save_dir, 'net_g_%d.npz' % epoch)
        # net_d_name_e = os.path.join(save_dir, 'net_d_%d.npz' % epoch)
        # tl.files.save_npz(net_cnn.all_params, name=net_c_name_e, sess=sess)
        # tl.files.save_npz(net_rnn.all_params, name=net_e_name_e, sess=sess)
        # tl.files.save_npz(net_g.all_params, name=net_g_name_e, sess=sess)
        # tl.files.save_npz(net_d.all_params, name=net_d_name_e, sess=sess)

        # tl.visualize.images2d(images=img_gen, second=0.01, saveable=True, name='temp_generate_%d' % epoch)#, dtype=np.uint8)
























































#
