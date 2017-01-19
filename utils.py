import tensorflow as tf
import os
import random
import scipy
import numpy as np

""" The functions here will be merged into TensorLayer after finishing this project.
"""

#files
def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.

    Parameters
    ----------
    path : a string or None
        A folder path.
    """
    return [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

#utils
def print_dict(dictionary={}):
    """Print all keys and items in a dictionary.
    """
    for key, value in dictionary.iteritems():
        print("key: %s  value: %s" % (str(key), str(value)))

#prepro ?
def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.

    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]



## Save images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def save_images(images, size, image_path):
    return imsave(images, size, image_path)


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














#
