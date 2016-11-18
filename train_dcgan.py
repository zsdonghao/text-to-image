import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from glob import glob
from random import shuffle

from utils import *
import re
import nltk

from model import *

pp = pprint.PrettyPrinter()

"""
TensorLayer implementation of DCGAN to generate image.

Usage : see README.md
"""

flags = tf.app.flags
flags.DEFINE_integer("epoch", 500, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 100, "The interval of generating sample. [100]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "flower", "The name of dataset [flower, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

# def inverse_transform(images):
#     return (images+1.)/2.

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def save_images(images, size, image_path):
    return imsave(images, size, image_path)
    # return imsave(inverse_transform(images), size, image_path)

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

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    z_dim = 100

    with tf.device("/gpu:0"):
        z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
        real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')

        # z --> generator for training
        net_g, _ = generator_dcgan(z, is_train=True, reuse=False)
        # generated fake images --> discriminator
        net_d, d_logits = discriminator_dcgan(net_g.outputs, is_train=True, reuse=False)
        # real images --> discriminator
        net_d2, d2_logits = discriminator_dcgan(real_images, is_train=True, reuse=True)
        # sample_z --> generator for evaluation, set is_train to False
        # so that BatchNormLayer behave differently
        net_g2, _ = generator_dcgan(z, is_train=False, reuse=True)

        # cost for updating discriminator and generator
        # discriminator: real images are labelled as 1
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d2_logits, tf.ones_like(d2_logits)))    # real == 1
        # discriminator: images from generator (fake) are labelled as 0
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits, tf.zeros_like(d_logits)))     # fake == 0
        d_loss = d_loss_real + d_loss_fake
        # generator: try to make the the fake images look real (1)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits, tf.ones_like(d_logits)))

        # trainable parameters for updating discriminator and generator
        g_vars = net_g.all_params   # only updates the generator
        d_vars = net_d.all_params   # only updates the discriminator

        net_g.print_params(False)
        print("---------------")
        net_d.print_params(False)

        # optimizers for updating discriminator and generator
        d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(g_loss, var_list=g_vars)

    sess=tf.Session()
    sess.run(tf.initialize_all_variables())

    # load checkpoints
    print("[*] Loading checkpoints...")
    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')
    if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
        print("[!] Loading checkpoints failed!")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_d_loaded_params = tl.files.load_npz(name=net_d_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        tl.files.assign_params(sess, net_d_loaded_params, net_d)
        print("[*] Loading checkpoints SUCCESS!")

    sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

    iter_counter = 0
    for epoch in range(FLAGS.epoch):
        idexs = get_random_int(min=0, max=n_captions-1, number=FLAGS.batch_size)
        sample_images = images[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
        print("[*]Sample images updated!")

        batch_idxs = int(n_images / FLAGS.batch_size)

        for idx in xrange(0, batch_idxs):
            idexs = get_random_int(min=0, max=n_captions-1, number=FLAGS.batch_size)
            batch_images = images[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
            batch_images = threading_data(batch_images, prepro_img, mode='train')
            batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
            start_time = time.time()
            # updates the discriminator
            errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images })
            # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, FLAGS.epoch, idx, batch_idxs,
                        time.time() - start_time, errD, errG))
            sys.stdout.flush()

            iter_counter += 1
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                # generate and visualize generated images
                img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
                '''
                img255 = (np.array(img) + 1) / 2 * 255
                tl.visualize.images2d(images=img255, second=0, saveable=True,
                                name='./{}/train_{:02d}_{:04d}'.format(FLAGS.sample_dir, epoch, idx), dtype=None, fig_idx=2838)
                '''
                img = threading_data(img, prepro_img, mode='rescale')
                save_images(img, [8, 8],
                            './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir+'/'+FLAGS.dataset+'_dcgan', epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))
                sys.stdout.flush()

            if np.mod(iter_counter, FLAGS.save_step) == 0:
                # save current network parameters
                print("[*] Saving checkpoints...")
                img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
                model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
                save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # the latest version location
                net_g_name = os.path.join(save_dir, 'net_g.npz')
                net_d_name = os.path.join(save_dir, 'net_d.npz')
                # this version is for future re-check and visualization analysis
                net_g_iter_name = os.path.join(save_dir, 'net_g_%d.npz' % iter_counter)
                net_d_iter_name = os.path.join(save_dir, 'net_d_%d.npz' % iter_counter)
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                tl.files.save_npz(net_g.all_params, name=net_g_iter_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_iter_name, sess=sess)
                print("[*] Saving checkpoints SUCCESS!")


if __name__ == '__main__':
    tf.app.run()
