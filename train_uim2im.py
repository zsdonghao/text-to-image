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


is_deep = True
if is_deep:
    # generator_txt2img = generator_txt2img_deep
    cnn_encoder = cnn_encoder_deep # use shallow cnn for text-image mapping, deep cnn for projection

def change_id(sentences, id_list=[], target_id=0):
    b_sentences = copy.deepcopy(sentences)
    for i, sen in enumerate(b_sentences):
        for j, w in enumerate(sen):
            if w in id_list:
                b_sentences[i][j] = target_id
                break   # only change one id in one sentence
    return b_sentences

def main_train_stackGAN():
    pass


def main_train_imageEncoder():
    # flower dataset
    # no deep        3000: 0.8; 8000: 0.6; 20000: 0.5; 800000: 0.16
    # deep G D E     3000: 0.8; 6000: 0.8; 10000:0.78; 20000: 0.75
    # deep E         1000: 0.6; 3000: 0.5; 6000: 0.48  20000: 0.38 min(0.34)
    t_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    net_rnn = rnn_embed(t_caption, is_train=False, reuse=False, return_embed=False)
    net_g, _ = generator_txt2img(t_z,
                    net_rnn,
                    is_train=False, reuse=False)

    net_p = cnn_encoder(net_g.outputs, is_train=True, reuse=False, name="image_encoder")

    # net_g1, _ = generator_txt2img(net_p.outputs,    # mse of x and x_z
    #                 net_rnn,
    #                 is_train=False, reuse=True)

    # for evaluation
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

        if (step != 0) and (step % 1000) == 0:
            b_images = sess.run(net_g3.outputs, feed_dict={
                                    t_caption : b_caption,
                                    t_z : b_z})

            gen_images = sess.run(net_g2.outputs, feed_dict={
                                    t_caption : b_caption,
                                    t_z : b_z})

            print('b_images', np.min(b_images), np.max(b_images))
            print('gen_images', np.min(gen_images), np.max(gen_images))

            print("[*] Sampling images")
            combine_and_save_image_sets([b_images, gen_images], 'samples/step2')

            print("[*] Saving Model")
            tl.files.save_npz(net_p.all_params, name=net_p_name, sess=sess)
            # tl.files.save_npz(net_p.all_params, name=net_p_name + "_" + str(step), sess=sess)
            print("[*] Model p(encoder) saved")

def main_translation():
    t_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'input_image')
    t_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='input_caption')
    # t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')
    # t_caption_p = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='caption_input_p')  #

    net_p = cnn_encoder(t_image, is_train=False, reuse=False, name="image_encoder")
    net_rnn = rnn_embed(t_caption, is_train=False, reuse=False, return_embed=False)
    net_g, _ = generator_txt2img(net_p.outputs, # image --> image
                    net_rnn,
                    is_train=False, reuse=False)

    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')   # debug, z --> image
    net_g2, _ = generator_txt2img(t_z,
                    net_rnn,
                    is_train=False, reuse=True)        # debug

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
        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            # # idexs = list(range(0, batch_size*n_captions_per_image, n_captions_per_image))   #
        b_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]   # real image
        b_images = threading_data(b_images, prepro_img, mode='translation')                                       # real image
        b_caption = captions_ids_train[idexs]   # for debug sample_sentence = b_caption
        b_caption = tl.prepro.pad_sequences(b_caption, padding='post') # for debug sample_sentence = b_caption

        # b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)    # use fake image
        # b_images = sess.run(net_g2.outputs, feed_dict={                                             # use fake image
        #                                 t_z : b_z,                                                  # use fake image
        #                                 t_caption : b_caption,                                      # use fake image
        #                                 })                                                          # use fake image

        save_images(b_images, [8, 8], 'samples/step3/source_{:02d}.png'.format(i))
        sample_sentence = change_id(b_caption, color_ids, vocab.word_to_id("blue"))
        # sample_sentence[0] = [vocab.word_to_id("blue")]
        # sample_sentence = b_caption                                           # don't change the sentences
        for idx, caption in enumerate(b_caption):
            print("%d-%d: source: %s" % (i, idx, [vocab.id_to_word(word) for word in caption]))
            print("%d-%d: target: %s" % (i, idx, [vocab.id_to_word(word) for word in sample_sentence[idx]]))
        # exit()

        # sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post') # for debug sample_sentence = b_caption

        gen_img = sess.run(net_g.outputs, feed_dict={
                                        t_image : b_images,
                                        t_caption : sample_sentence,
                                        })

        print(np.min(b_images), np.max(b_images))
        print(np.min(gen_img), np.max(gen_img))

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

    parser.add_argument('--train_step', type=str, default="imageEncoder",
                       help='Step of the training : translation, imageEncoder')

    # parser.add_argument('--retrain', type=int, default=0,
    #                    help='Set 0 for using pre-trained model, 1 for retraining the model')

    args = parser.parse_args()

    # FLAGS.retrain = args.retrain == 1

    if args.train_step == "imageEncoder":
        main_train_imageEncoder()

    elif args.train_step == "translation":
        main_translation()




















































#
