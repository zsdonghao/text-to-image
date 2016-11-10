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
    #  save_images(img, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
    return imsave(inverse_transform(images), size, image_path)

# layers helper functon
# def list_remove_repeat(l=[]):
#     """Remove the repeated items in a list, and return the processed list.
#
#     Parameters
#     ----------
#     l : a list
#
#     Examples
#     ---------
#     >>> l = [2, 3, 4, 2, 3]
#     >>> l = list_remove_repeat(l)
#     ... [2, 3, 4]
#     """
#     l2 = []
#     [l2.append(i) for i in l if not i in l2]
#     return l2


#layers Name Scope Functions
# print_all_variables
# def get_variables_with_name(name, train_only=True, printable=False):
#     """Get variable list by a given name scope.
#
#     Examples
#     ---------
#     >>> dense_vars = get_variable_with_name('dense', True, True)
#     """
#     print("  Get variables with %s" % name)
#     # t_vars = tf.trainable_variables()
#     t_vars = tf.trainable_variables() if train_only else tf.all_variables()
#     d_vars = [var for var in t_vars if name in var.name]
#     if printable:
#         for idx, v in enumerate(d_vars):
#             print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
#     return d_vars





















#
