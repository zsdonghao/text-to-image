import tensorflow as tf
import os
import random

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
    """Print all key and item of a dictionary.
    """
    for key, value in dictionary.iteritems():
        print("key: %s  value: %s" % (str(key), str(value)))


#prepro ?
def generate_random_int(min=0, max=10, number=5):
    """Return a list of random integer by given range and the number.

    Examples
    ---------
    >>> r = generate_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]



#utils ?
def list_remove_repeat(l=[]):
    """Remove the repeated items in a list, and return the processed list.

    Parameters
    ----------
    l : a list

    Examples
    ---------
    >>> l = [2, 3, 4, 2, 3]
    >>> l = list_remove_repeat(l)
    ... [2, 3, 4]
    """
    l2 = []
    [l2.append(i) for i in l if not i in l2]
    return l2


#layers Name Scope Functions
# print_all_variables
def get_variable_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.
    """
    print("  Get variables with %s" % name)
    # t_vars = tf.trainable_variables()
    t_vars = tf.trainable_variables() if train_only else tf.all_variables()
    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars





















#
