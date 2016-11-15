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
def generate_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.

    Examples
    ---------
    >>> r = generate_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]

from tensorlayer.prepro import *
def prepro_img(x, mode=None):
    if mode=='train':   # [0, 255] --> (-1, 1), random flip left and right
        x = x / (255. / 2.)
        x = x - 1.
        x = flip_axis(x, axis=1, is_random=True)
    elif mode=='rescale':  # (-1, 1) --> (0, 1)
        x = (x + 1.) / 2.
    elif mode=='debug':
        x = flip_axis(x, axis=1, is_random=True)
    else:
        raise Exception("Not support : %s" % mode)
    return x

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

















#
