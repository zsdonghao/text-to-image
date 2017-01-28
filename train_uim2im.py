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
import copy

from utils import *
from model import *

import argparse

## Load Oxford 102 flowers dataset
from data_loader import *


# is_deep = True
# if is_deep:
# cnn_encoder = cnn_encoder_deep # use shallow cnn for text-image mapping, deep cnn for projection


def change_id(sentences, id_list=[], target_id=0):
    b_sentences = copy.deepcopy(sentences)
    for i, sen in enumerate(b_sentences):
        for j, w in enumerate(sen):
            if w in id_list:
                b_sentences[i][j] = target_id
                break   # only change one id in one sentence
    return b_sentences

def main_train_stackGAN():
    image_size = 256
    images_train = images_train_256
    stackG = stackG_256
    stackD = stackD_256
    # print(images_train.shape)

    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    # t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=False, return_embed=False)

    net_fake_image_g1, _ = generator_txt2img(t_z,
                    net_rnn,
                    is_train=False, reuse=False)

    net_fake_image, _ = stackG(net_fake_image_g1.outputs,
                    net_rnn,
                    is_train=True, reuse=False)
    net_d, disc_fake_image_logits = stackD(
                    net_fake_image.outputs,
                    net_rnn,
                    is_train=True, reuse=False)
    _, disc_real_image_logits = stackD(
                    t_real_image,
                    net_rnn,
                    is_train=True, reuse=True)
    _, disc_wrong_image_logits = stackD( # CLS
                    t_wrong_image,
                    net_rnn,
                    is_train=True, reuse=True)

    ## testing inference for txt2img
    net_gII, _ = stackG(net_fake_image_g1.outputs,
                    rnn_embed(t_real_caption, is_train=False, reuse=True, return_embed=False),
                    is_train=False, reuse=True)

    # net_gII.print_params(False)
    # exit()

    d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits, tf.ones_like(disc_real_image_logits)))
    d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image_logits)))    # for CLS, if set it to zero, it is the same with normal DCGAN
    d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits)))

    d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits))) # real == 1, fake == 0

    d_vars = tl.layers.get_variables_with_name('stackD', True, True)
    g_vars = tl.layers.get_variables_with_name('stackG', True, True)

    lr = 0.0002
    lr_decay = 0.5
    decay_every = 100
    beta1 = 0.5

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars )
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars )

    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)

    # load the latest checkpoints
    save_dir = "checkpoint"
    # os.system("mkdir checkpoint/step2")
    os.system("mkdir samples/stackGAN")
    net_e_name = os.path.join(save_dir, 'net_e.npz')
    net_g_name = os.path.join(save_dir, 'net_g.npz')

    if not os.path.exists(net_e_name):
        print("[!] Loading RNN checkpoint failed!")
    else:
        net_e_loaded_params = tl.files.load_npz(name=net_e_name)
        tl.files.assign_params(sess, net_e_loaded_params, net_rnn)
        del net_e_loaded_params
        print("[*] Loading RNN checkpoint SUCCESS!")

    if not os.path.exists(net_g_name):
        print("[!] Loading G I checkpoint failed!")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_fake_image_g1)
        del net_g_loaded_params
        print("[*] Loading G I checkpoint SUCCESS!")

    net_stackG_name = os.path.join(save_dir, 'net_stackG.npz')
    net_stackD_name = os.path.join(save_dir, 'net_stackD.npz')
    if not os.path.exists(net_stackG_name):
        print("[!] Loading G II checkpoint failed!")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_stackG_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_gII)
        del net_g_loaded_params
        print("[*] Loading G II checkpoint SUCCESS!")
    if not os.path.exists(net_stackD_name):
        print("[!] Loading D II checkpoint failed!")
            # try:
            #     # as the architecture of D II equal to D I, you can initialize D II by using D I parameters.
            #     net_d_loaded_params = tl.files.load_npz(name="checkpoint/net_d.npz")
            #     tl.files.assign_params(sess, net_d_loaded_params, net_d)
            #     print("[*] Loading D II from D I SUCCESS!")
            # except:
            #     print("[*] Loading D II from D I failed!")
    else:
        net_d_loaded_params = tl.files.load_npz(name=net_stackD_name)
        tl.files.assign_params(sess, net_d_loaded_params, net_d)
        del net_d_loaded_params
        print("[*] Loading D II checkpoint SUCCESS!")

        # # as the architecture of D II equal to D I, you can initialize D II by using D I parameters.
        # net_d_loaded_params = tl.files.load_npz(name="checkpoint/net_d.npz")
        # tl.files.assign_params(sess, net_d_loaded_params, net_d)
        # print("[*] Loading D II from D I SUCCESS!")

    sample_size = batch_size
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
    sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/8) + \
                      ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/8) + \
                      ["the petals on this flower are white with a yellow center"] * int(sample_size/8) + \
                      ["this flower has a lot of small round pink petals."] * int(sample_size/8) + \
                      ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/8) + \
                      ["the flower has yellow petals and the center of it is brown."] * int(sample_size/8) + \
                      ["this flower has petals that are blue and white."] * int(sample_size/8) +\
                      ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/8)

    for i, sentence in enumerate(sample_sentence):
        print("seed: %s" % sentence)
        sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]    # add END_ID
    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    ## gI outputs for comparsion with gII outputs
    img_gen, _ = sess.run([net_fake_image_g1.outputs, net_rnn.outputs],
                                feed_dict={
                                t_real_caption : sample_sentence,
                                t_z : sample_seed})
    save_images(img_gen, [8, 8], '{}/stackGAN/train__g1.png'.format('samples'))

    # debug
    # img_gen, rnn_out = sess.run([net_gII.outputs, net_rnn.outputs],
    #                             feed_dict={
    #                             t_real_caption : sample_sentence,
    #                             t_z : sample_seed})
    # save_images(img_gen, [8, 8], '{}/stackGAN/train__g2.png'.format('samples'))
    # exit()

    n_epoch = 600   # 600 when pre-trained rnn
    print_freq = 1
    n_batch_epoch = int(n_images / batch_size)
    for epoch in range(0, n_epoch+1):
        start_time = time.time()

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

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

            ## updates D
            b_real_images = threading_data(b_real_images, prepro_img, mode='train_stackGAN')   # [0, 255] --> [-1, 1]
            b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train_stackGAN')
            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                            t_real_image : b_real_images,
                            t_wrong_image : b_wrong_images,
                            t_real_caption : b_real_caption,
                            t_z : b_z})

            ## updates G
            for _ in range(1):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={
                                t_real_caption : b_real_caption,
                                t_z : b_z})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG))

        if (epoch + 1) % print_freq == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))

            img_gen, rnn_out = sess.run([net_gII.outputs, net_rnn.outputs],
                                        feed_dict={
                                        t_real_caption : sample_sentence,  # remove if DCGAN only
                                        t_z : sample_seed})

            # print('rnn:', np.min(rnn_out[0]), np.max(rnn_out[0]))   # -1.4121389, 1.4108921
            # print('real:', b_real_images[0].shape, np.min(b_real_images[0]), np.max(b_real_images[0]))
            # print('wrong:', b_wrong_images[0].shape, np.min(b_wrong_images[0]), np.max(b_wrong_images[0]))
            # print('generate:', img_gen[0].shape, np.min(img_gen[0]), np.max(img_gen[0]))
            # img_gen = threading_data(img_gen, prepro_img, mode='rescale')

            # if need_256:
            # 256x256 image is very large, resize it before save it
            img_gen = threading_data(img_gen, imresize, size=[64, 64], interp='bilinear')

            save_images(img_gen, [8, 8], '{}/stackGAN/train_{:02d}.png'.format('samples', epoch))

        # tl.files.save_npz(net_gII.all_params, name=net_stackG_name, sess=sess)
        # tl.files.save_npz(net_d.all_params, name=net_stackD_name, sess=sess)
        # print("[*] Saving stackG, stackD checkpoints SUCCESS!")

        if (epoch != 0) and (epoch % 5) == 0:
            tl.files.save_npz(net_gII.all_params, name=net_stackG_name, sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_stackD_name, sess=sess)
            print("[*] Saving stackG, stackD checkpoints SUCCESS!")

        if (epoch != 0) and (epoch % 100) == 0:
            net_stackG_name_e = os.path.join(save_dir, 'net_stackG_%d.npz' % epoch)
            net_stackD_name_e = os.path.join(save_dir, 'net_stackD_%d.npz' % epoch)
            tl.files.save_npz(net_gII.all_params, name=net_stackG_name_e, sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_stackD_name_e, sess=sess)


def main_train_imageEncoder():
    # flower dataset
    # no deep        1000: 0.8; 8000: 0.6; 20000: 0.5; 800000: 0.16
    # deep G D E     3000: 0.8; 6000: 0.8; 10000:0.78; 20000: 0.75
    # deep E         1000: 0.4;
    # stackG deep E  1000: 0.75;
    # E_256,         2000: 0.87 6000: 0.8 10000: 0.77 13172: 0.76
    is_stackGAN = True # use stackGAN and use E with 256x256x3 input
    if is_stackGAN:
        stackG = stackG_256
        stackD = stackD_256
        cnn_encoder = cnn_encoder_256

    t_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    net_rnn = rnn_embed(t_caption, is_train=False, reuse=False, return_embed=False)
    net_g, _ = generator_txt2img(t_z,
                    net_rnn,
                    is_train=False, reuse=False)
    if is_stackGAN:
        net_gII, _ = stackG(net_g.outputs,
                        net_rnn,
                        is_train=False, reuse=False)
        net_p = cnn_encoder(net_gII.outputs, is_train=True, reuse=False, name="image_encoder")
        # ## downsampling from 256 to 64
        # net_gII = DownSampling2dLayer(net_gII, size=[64, 64], is_scale=False, method=0, name='stackG_output_downsampling') # 0: bilinear 1: nearest
        # net_p = cnn_encoder(net_gII.outputs, is_train=True, reuse=False, name="image_encoder")
    else:
        net_p = cnn_encoder(net_g.outputs, is_train=True, reuse=False, name="image_encoder")

    # net_g1, _ = generator_txt2img(net_p.outputs,    # mse of x and x_z
    #                 net_rnn,
    #                 is_train=False, reuse=True)

    # for evaluation
    if is_stackGAN:
        net_g2, _ = generator_txt2img(net_p.outputs,    # for evaluation, generate from P
                        net_rnn,
                        is_train=False, reuse=True)
        net_g2, _ = stackG(net_g2.outputs,
                        net_rnn,
                        is_train=False, reuse=True)
        net_g3, _ = generator_txt2img(t_z,              # for evaluation, generate from z
                        net_rnn,
                        is_train=False, reuse=True)
        net_g3, _ = stackG(net_g3.outputs,
                        net_rnn,
                        is_train=False, reuse=True)
    else:
        net_g2, _ = generator_txt2img(net_p.outputs,    # for evaluation, generate from P
                        net_rnn,
                        is_train=False, reuse=True)
        net_g3, _ = generator_txt2img(t_z,              # for evaluation, generate from z
                        net_rnn,
                        is_train=False, reuse=True)

    lr = 0.0002
    lr_decay = 0.5
    decay_every = 5000
    beta1 = 0.5
    n_step = 100000 #* 100
    sample_size = batch_size

    train_vars = tl.layers.get_variables_with_name('image_encoder', True, True)
    loss = tf.reduce_mean( tf.square( tf.sub( net_p.outputs, t_z) ))  # +   tf.reduce_mean( tf.square( tf.sub( net_g1.outputs, net_g.outputs) ))
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=train_vars)

    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)

    # load the latest checkpoints
    save_dir = "checkpoint"
    # os.system("mkdir checkpoint/step2")
    os.system("mkdir samples/step2")
    net_e_name = os.path.join(save_dir, 'net_e.npz')
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_p_name = os.path.join(save_dir, 'net_p.npz')

    if not os.path.exists(net_e_name):
        print("[!] Loading RNN checkpoint failed!")
    else:
        net_e_loaded_params = tl.files.load_npz(name=net_e_name)
        tl.files.assign_params(sess, net_e_loaded_params, net_rnn)
        print("[*] Loading RNN checkpoint SUCCESS!")

    if not os.path.exists(net_g_name):
        print("[!] Loading G checkpoint failed!")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        print("[*] Loading G checkpoint SUCCESS!")

    if is_stackGAN:
        net_stackG_name = os.path.join(save_dir, 'net_stackG.npz')
        if not os.path.exists(net_stackG_name):
            print("[!] Loading G II checkpoint failed!")
        else:
            net_g_loaded_params = tl.files.load_npz(name=net_stackG_name)
            tl.files.assign_params(sess, net_g_loaded_params, net_gII)
            print("[*] Loading G II checkpoint SUCCESS!")


    if True:
        if not os.path.exists(net_p_name):
            print("[!] Loading Encoder checkpoint failed!")
        else:
            net_p_loaded_params = tl.files.load_npz(name=net_p_name)
            tl.files.assign_params(sess, net_p_loaded_params, net_p)
            print("[*] Loading Encoder checkpoint SUCCESS!")

    for step in range(n_step):
        ## decay lr
        if step !=0 and (step % decay_every == 0):
            new_lr_decay = lr_decay ** (step // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif step == 0:
            log = " ** init lr: %f  decay_every: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        # ## get matched text
        # idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
        # b_real_caption = captions_ids_train[idexs]                                                                      # remove if DCGAN only
        # b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')     # matched text  (64, any)    # remove if DCGAN only
        # ## get real image
        # b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]   # real images   (64, 64, 64, 3)
        # ## get wrong caption
        # idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
        # b_wrong_caption = captions_ids[idexs]
        # b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')                                    # mismatched text
        # ## get wrong image
        # idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)        # remove if DCGAN only
        # b_wrong_images = images_train[idexs2]                                               # remove if DCGAN only
        # ## get noise
        # b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
        b_caption = captions_ids_train[idexs]
        b_caption = tl.prepro.pad_sequences(b_caption, padding='post')

        # b_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')] #
        # b_real_images = threading_data(b_real_images, prepro_img, mode='rescale')

        b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

        _, err = sess.run([train_op, loss], feed_dict={
                                    t_caption : b_caption,
                                    t_z : b_z,
                                        })

        print("step[{}/{}] loss:{}".format(step, n_step, err))


        # b_images = sess.run(net_gII.outputs, feed_dict={ # debug
        #                         t_caption : b_caption,
        #                         t_z : b_z})
        # gen_images = sess.run(net_g2.outputs, feed_dict={
        #                         t_caption : b_caption,
        #                         t_z : b_z})
        # print('b_images', np.min(b_images), np.max(b_images))
        # print('gen_images', np.min(gen_images), np.max(gen_images))
        # print("[*] Sampling images")
        # combine_and_save_image_sets([b_images, gen_images], 'samples/step2')
        # exit()


        if (step != 0) and (step % 1000) == 0:
            b_images = sess.run(net_g3.outputs, feed_dict={
                                    t_caption : b_caption,
                                    t_z : b_z})

            gen_images = sess.run(net_g2.outputs, feed_dict={
                                    t_caption : b_caption,
                                    t_z : b_z})

            print('b_images', np.min(b_images), np.max(b_images))
            print('gen_images', np.min(gen_images), np.max(gen_images))

            if is_stackGAN:
                # 256x256 images are very large, reduce the size before saving them
                b_images = threading_data(b_images, imresize, size=[64, 64], interp='bilinear')
                gen_images = threading_data(gen_images, imresize, size=[64, 64], interp='bilinear')

            print("[*] Sampling images")
            combine_and_save_image_sets([b_images, gen_images], 'samples/step2')

            print("[*] Saving Model")
            tl.files.save_npz(net_p.all_params, name=net_p_name, sess=sess)
            # tl.files.save_npz(net_p.all_params, name=net_p_name + "_" + str(step), sess=sess)


def main_translation():
    is_stackGAN = True # use stackGAN and use E with 256x256x3 input, otherwise, 64x64x3 as input
    if is_stackGAN:
        image_size = 256
        stackG = stackG_256
        cnn_encoder = cnn_encoder_256
        images_test = images_test_256
    else:
        image_size = 64
        import model
        cnn_encoder = model.cnn_encoder
        # cnn_encoder = cnn_encoder

    t_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'input_image')
    t_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='input_caption')
    # t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')
    # t_caption_p = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='caption_input_p')  #

    net_p = cnn_encoder(t_image, is_train=False, reuse=False, name="image_encoder")
    net_rnn = rnn_embed(t_caption, is_train=False, reuse=False, return_embed=False)
    net_g, _ = generator_txt2img(net_p.outputs, # image --> image
                    net_rnn,
                    is_train=False, reuse=False)

    if is_stackGAN:
        net_gII, _ = stackG(net_g.outputs,
                        net_rnn,
                        is_train=False, reuse=False)

    # use fake image as input
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')   # debug, z --> image
    net_g2, _ = generator_txt2img(t_z,
                    net_rnn,
                    is_train=False, reuse=True)        # debug
    if is_stackGAN:
        net_g2, _ = stackG(net_g2.outputs,
                        net_rnn,
                        is_train=False, reuse=True)

    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)

    # load the latest checkpoints
    save_dir = "checkpoint"
    # os.system("mkdir checkpoint/step3")
    os.system("mkdir samples/step3")
    net_e_name = os.path.join(save_dir, 'net_e.npz')
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_p_name = os.path.join(save_dir, 'net_p.npz')

    # load generator, RNN and Encoder
    net_e_loaded_params = tl.files.load_npz(name=net_e_name)
    tl.files.assign_params(sess, net_e_loaded_params, net_rnn)
    net_g_loaded_params = tl.files.load_npz(name=net_g_name)
    tl.files.assign_params(sess, net_g_loaded_params, net_g)
    net_p_loaded_params = tl.files.load_npz(name=net_p_name)
    tl.files.assign_params(sess, net_p_loaded_params, net_p)
    if is_stackGAN:
        net_stackG_name = os.path.join(save_dir, 'net_stackG.npz')
        net_stackG_loaded_params = tl.files.load_npz(name=net_stackG_name)
        tl.files.assign_params(sess, net_stackG_loaded_params, net_gII)

    sample_size = batch_size
    # sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/8) + \
    #                   ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/8) + \
    #                   ["the petals on this flower are white with a yellow center"] * int(sample_size/8) + \
    #                   ["this flower has a lot of small round pink petals."] * int(sample_size/8) + \
    #                   ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/8) + \
    #                   ["the flower has yellow petals and the center of it is brown."] * int(sample_size/8) + \
    #                   ["this flower has petals that are blue and white."] * int(sample_size/8) +\
    #                   ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/8)
    #
    # ## for idexs = list(range(batch_size))
    # # sample_sentence = ["prominent white stigma petals are white in color"] * int(sample_size)   # 1st image in dataset
    # # sample_sentence = ["this flower is red in color, with petals that are oval shaped"] * int(sample_size)  # 64th image in dataset
    #
    # # sample_sentence = captions_ids_test[0:sample_size]
    # for i, sentence in enumerate(sample_sentence):
    #     print("seed: %s" % sentence)
    #     sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]    # add END_ID
    #     # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
    #     # print(sample_sentence[i])
    # sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    color_ids = [vocab.word_to_id(w) for w in ["red", "green", "yellow", "blue", "white", "pink", "purple", "orange", "black", "orange-yellow", "brown", "pink-lavendar", "lavender"]]

    for i in range(1):
        idexs = get_random_int(min=0, max=n_captions_test-1, number=batch_size)
            # # idexs = list(range(0, batch_size*n_captions_per_image, n_captions_per_image))   #
        b_images = images_test[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]   # real image
        b_images = threading_data(b_images, prepro_img, mode='translation')                                       # real image
        b_caption = captions_ids_test[idexs]   # for debug sample_sentence = b_caption
        b_caption = tl.prepro.pad_sequences(b_caption, padding='post') # for debug sample_sentence = b_caption

        # b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)    # use fake image
        # b_images = sess.run(net_g2.outputs, feed_dict={                                             # use fake image
        #                                 t_z : b_z,                                                  # use fake image
        #                                 t_caption : b_caption,                                      # use fake image
        #                                 })                                                          # use fake image


        sample_sentence = change_id(b_caption, color_ids, vocab.word_to_id("blue"))
        # sample_sentence[0] = [vocab.word_to_id("blue")]
        # sample_sentence = b_caption                                               # reconstruct from same sentences, test performance of reconstruction
        for idx, caption in enumerate(b_caption):
            print("%d-%d: source: %s" % (i, idx, [vocab.id_to_word(word) for word in caption]))
            print("%d-%d: target: %s" % (i, idx, [vocab.id_to_word(word) for word in sample_sentence[idx]]))
        # exit()

        # sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post') # for debug sample_sentence = b_caption
        if is_stackGAN:
            gen_img = sess.run(net_gII.outputs, feed_dict={
                                            t_image : b_images,
                                            t_caption : sample_sentence,
                                            })
            # 256x256 images are too large, resize to 64
            b_images = threading_data(b_images, imresize, size=[64, 64], interp='bilinear')
            gen_img = threading_data(gen_img, imresize, size=[64, 64], interp='bilinear')
        else:
            gen_img = sess.run(net_g.outputs, feed_dict={
                                            t_image : b_images,
                                            t_caption : sample_sentence,
                                            })

        # print(np.min(b_images), np.max(b_images))
        # print(np.min(gen_img), np.max(gen_img))
        save_images(b_images, [8, 8], 'samples/step3/source_{:02d}.png'.format(i))
        save_images(gen_img, [8, 8], 'samples/step3/translate_{:02d}.png'.format(i))
        # print("Translate completed {}".format(i))

        # debug
        # b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
        # gen_img2 = sess.run(net_g2.outputs, feed_dict={
        #                                 t_z : b_z,
        #                                 t_caption : b_caption,
        #                                 })
        # save_images(gen_img2, [8, 8], 'samples/step3/debug_{:02d}.png'.format(i))
        print("Translate completed {}".format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_step', type=str, default="stackGAN",
                       help='Step of the training : stackGAN, imageEncoder, translation')

        # parser.add_argument('--retrain', type=int, default=0,
        #                    help='Set 0 for using pre-trained model, 1 for retraining the model')

    args = parser.parse_args()

        # FLAGS.retrain = args.retrain == 1

    if args.train_step == "stackGAN":
        main_train_stackGAN()

    elif args.train_step == "imageEncoder":
        main_train_imageEncoder()

    elif args.train_step == "translation":
        main_translation()






















































#
