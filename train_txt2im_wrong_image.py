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
from model import *


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
    """
    images.shape = [8000, 64, 64, 3]
    captions_ids = [80000, any]
    """
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

    print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

    captions_ids_train, captions_ids_test = captions_ids[: 8000*n_captions_per_image], captions_ids[8000*n_captions_per_image :]
    images_train, images_test = images[:8000], images[8000:]
    n_images_train = len(images_train)
    n_images_test = len(images_test)
    n_captions_train = len(captions_ids_train)
    n_captions_test = len(captions_ids_test)
    print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
    print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))

    ## check test image
    # idexs = get_random_int(min=0, max=n_captions_test-1, number=64)
    # temp_test_capt = captions_ids_test[idexs]
    # for idx, ids in enumerate(temp_test_capt):
    #     print("%d %s" % (idx, [vocab.id_to_word(id) for id in ids]))
    # temp_test_img = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    # save_images(temp_test_img, [8, 8], 'temp_test_img.png')
    # exit()

    # ## check the first example
    # tl.visualize.frame(I=images[0], second=5, saveable=True, name='temp', cmap=None)
    # for cap in captions_dict[1]:
    #     print(cap)
    # print(captions_ids[0:10])
    # for ids in captions_ids[0:10]:
    #     print([vocab.id_to_word(id) for id in ids])
    # print_dict(captions_dict)

    # ## generate a random batch
    # idexs = get_random_int(0, n_captions, batch_size)
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

###======================== DEFIINE MODEL ===================================###
## define data augmentation method
from tensorlayer.prepro import *
def prepro_img(x, mode=None):
    if mode=='train':
    # rescale [0, 255] --> (-1, 1), random flip, crop, rotate
    #   paper 5.1: During mini-batch selection for training we randomly pick
    #   an image view (e.g. crop, flip) of the image and one of the captions
    # flip, rotate, crop, resize : https://github.com/reedscot/icml2016/blob/master/data/donkey_folder_coco.lua
    # flip : https://github.com/paarthneekhara/text-to-image/blob/master/Utils/image_processing.py
        # x = flip_axis(x, axis=1, is_random=True)
        # x = rotation(x, rg=16, is_random=True, fill_mode='nearest')
        # x = crop(x, wrg=50, hrg=50, is_random=True)
        # x = imresize(x, size=[64, 64], interp='bilinear', mode=None)
        x = x / (255. / 2.)
        x = x - 1.
    elif mode=='rescale':
    # rescale (-1, 1) --> (0, 1) for display
        x = (x + 1.) / 2.
    elif mode=='debug':
        x = flip_axis(x, axis=1, is_random=False)
        # x = rotation(x, rg=16, is_random=False, fill_mode='nearest')
        # x = crop(x, wrg=50, hrg=50, is_random=True)
        # x = imresize(x, size=[64, 64], interp='bilinear', mode=None)
        x = x / 255.
    else:
        raise Exception("Not support : %s" % mode)
    return x

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
t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

## training inference for training DCGAN
# from dcgan_model import *
# net_fake_image, _ = generator_dcgan(t_z, is_train=True, reuse=False)
# _, disc_fake_image_logits = discriminator_dcgan(net_fake_image.outputs, is_train=True, reuse=False)
# _, disc_real_image_logits = discriminator_dcgan(t_real_image, is_train=True, reuse=True)
## training inference for txt2img
is_train_rnn = True
net_rnn = rnn_embed(t_real_caption, is_train=is_train_rnn, reuse=False)   # remove if DCGAN only
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
                rnn_embed(t_real_caption, is_train=False, reuse=True), # remove if DCGAN only
                is_train=False, reuse=True)

d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits, tf.ones_like(disc_real_image_logits)))
d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image_logits)))    # for CLS, if set it to zero, it is the same with normal DCGAN
d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits)))

d_loss = d_loss1 + d_loss2 + d_loss3

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits))) # real == 1, fake == 0

net_fake_image.print_params(False)
net_fake_image.print_layers()
# exit()

####======================== DEFINE TRAIN OPTS ==========================###
## Cost   real == 1, fake == 0
lr = 0.0002
beta1 = 0.5
n_g_batch = 2   # update G, x time per batch
e_vars = tl.layers.get_variables_with_name('rnn', True, True)           #  remove if DCGAN only
d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
g_vars = tl.layers.get_variables_with_name('generator', True, True)

## results
# update rnn in both D and G: 700 epochs lr=2e-4 d_loss: 0.09966065, g_loss: 3.55960941     Hao : don't update RNN and G together, G is cheat D
# update rnn only in G:
# update rnn only in D:
#   1000 epochs lr=2e-4 beta1=0.5           d_loss: 0.00363965, g_loss: 10.59220123  (at the begining, it work, but finially g_loss increase, RNN overfit?)
#   500  epochs lr=1e-4 beta1=0.9 rnn 512   d_loss: 0.03315957, g_loss: 5.69525194   (at 100 epochs, it is correct ! but sometime turn to incorrect color.)
#       rnn size 200 dp 0.9, don't work, all the same
# pre-trained RNN with D for 100 epoch, then don't update RNN anymore.
#   200 epochs lr=2e-4 beta=0.5 rnn 512 dp 0.5   d_loss: 0.40549859, g_loss: 6.31078005,        g_loss still increase
#   increase t_dim from 128 to 256      250 epoch d_loss: 0.01651493, g_loss: 11.65070343,      g_loss still increase
#   update G 3 times per batch          300 epoch d_loss: 0.35285434, g_loss: 4.30621910        g_loss still increase
#   change lr to 2e-5                   images under a caption look the same                    g_loss still increase
#   l2=1e-4, update G 5 times           images under a caption look the same
#   set net_rnn_embed(is_train=False) , no random filp          100 epoch: good, 200 epoch: images under a caption look the same, g_loss still increase
#   use wrong caption instead of wrong image,                   100 epoch: good, 200 epoch: images under a caption look the same, g_loss still increase

# grads = tf.gradients(d_loss, d_vars + e_vars)
# grads, _ = tf.clip_by_global_norm(tf.gradients(d_loss, d_vars + e_vars), 30)
# optimizer = tf.train.AdamOptimizer(1e-4, beta1=beta1)
# d_optim = optimizer.apply_gradients(zip(grads, d_vars + e_vars))
#
# grads = tf.gradients(g_loss, g_vars)
# grads, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars), 30)
# optimizer = tf.train.AdamOptimizer(1e-4, beta1=beta1)
# g_optim = optimizer.apply_gradients(zip(grads, g_vars))

d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(d_loss, var_list=d_vars )#+ e_vars)
g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(g_loss, var_list=g_vars )#+ e_vars)

###============================ TRAINING ====================================###
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

save_dir = "checkpoint"
if not os.path.exists(save_dir):
    print("[!] Folder (%s) is not exist, creating it ..." % save_dir)
    os.mkdir(save_dir)
# load the latest checkpoints
net_e_name = os.path.join(save_dir, 'net_e.npz')
net_g_name = os.path.join(save_dir, 'net_g.npz')
net_d_name = os.path.join(save_dir, 'net_d.npz')
if not os.path.exists(net_e_name):
    print("[!] Loading RNN checkpoints failed!")
else:
    net_e_loaded_params = tl.files.load_npz(name=net_e_name)
    tl.files.assign_params(sess, net_e_loaded_params, net_rnn)
    print("[*] Loading RNN checkpoints SUCCESS!")

# if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
#     print("[!] Loading G and D checkpoints failed!")
# else:
#     net_g_loaded_params = tl.files.load_npz(name=net_g_name)
#     net_d_loaded_params = tl.files.load_npz(name=net_d_name)
#     tl.files.assign_params(sess, net_g_loaded_params, net_g)
#     tl.files.assign_params(sess, net_d_loaded_params, net_d)
#     print("[*] Loading G and D checkpoints SUCCESS!")

# sess=tf.Session()
# tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
# sess.run(tf.initialize_all_variables())

## seed for generation, z and sentence ids
sample_size = batch_size
sample_seed = np.random.uniform(low=-1, high=1, size=(sample_size, z_dim)).astype(np.float32)        # paper said [0, 1]
# sample_sentence = ["this white and yellow flower have thin white petals and a round yellow stamen", \
#                     "the flower has petals that are bright pinkish purple with white stigma"] * 32
# sample_sentence = ["these flowers have petals that start off white in color and end in a dark purple towards the tips"] * 32 + \
#                     ["bright droopy yellow petals with burgundy streaks and a yellow stigma"] * 32
sample_sentence = ["these white flowers have petals that start off white in color and end in a white towards the tips",
                    "this yellow petals with burgundy streaks and a yellow stigma"] * 32
# sample_sentence = captions_ids_test[0:sample_size]
for i, sentence in enumerate(sample_sentence):
    print("seed: %s" % sentence)
    sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)]
    # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
    # print(sample_sentence[i])
sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

n_epoch = 1000   # 600 when pre-trained rnn
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
        # idexs = get_random_int(min=0, max=n_captions-1, number=batch_size)
        # b_wrong_caption = captions_ids[idexs]
        # b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')                                    # mismatched text
        ## get wrong image
        idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)        # remove if DCGAN only
        b_wrong_images = images_train[idexs2]                                               # remove if DCGAN only
        ## get noise
        b_z = np.random.uniform(low=-1, high=1, size=[batch_size, z_dim]).astype(np.float32)       # paper said [0, 1], but [-1, 1] is better
        ## check data
        # print(np.min(b_real_images), np.max(b_real_images), b_real_images.shape)    # [0, 1] (64, 64, 64, 3)
        # for i, seq in enumerate(b_real_caption):
        #     # print(seq)
        #     print(i, " ".join([vocab.id_to_word(id) for id in seq]))
        # save_images(b_real_images, [8, 8], 'real_image.png')
        # exit()

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

        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG))

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

    if epoch % 50 == 0:
        # tl.files.save_npz(net_rnn.all_params, name=net_e_name, sess=sess)
        # tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
        # tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
        # net_e_name_e = os.path.join(save_dir, 'net_e_%d.npz' % epoch)
        # net_g_name_e = os.path.join(save_dir, 'net_g_%d.npz' % epoch)
        # net_d_name_e = os.path.join(save_dir, 'net_d_%d.npz' % epoch)
        # tl.files.save_npz(net_rnn.all_params, name=net_e_name_e, sess=sess)
        # tl.files.save_npz(net_g.all_params, name=net_g_name_e, sess=sess)
        # tl.files.save_npz(net_d.all_params, name=net_d_name_e, sess=sess)
        print("[*] Saving checkpoints SUCCESS!")
        # tl.visualize.images2d(images=img_gen, second=0.01, saveable=True, name='temp_generate_%d' % epoch)#, dtype=np.uint8)
























































#
