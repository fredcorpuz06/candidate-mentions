'''Helper functions for managing files, folders
'''

import os
import re
import shutil
import random
import functools


def copytree(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d)
        else:
            # check if the file is modified then only we should copy
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


def repeat(func):
    '''Allows arbitrary function to repeat over varying number of inputs.'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 1:
            value = func(*args, **kwargs)
        else:
            value = [func(a, **kwargs) for a in args]
        return value

    return wrapper


@repeat
def create_folder(out_folder, *out_folders):
    '''Creates folders
    TO DO: assert valid folder

    Examples
    >>> create_folder("../data/faces-dataset/train")
    "../data/faces-dataset/train"
    >>> create_folder(
        "../data/faces-dataset/train", 
        "../data/faces-dataset/valid", 
        "../data/faces-dataset/sample", )
    ["../data/faces-dataset/train", "../data/faces-dataset/valid",
     "../data/faces-dataset/sample"]
    '''
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    return out_folder


def replace_spaces(my_string, substitute="_"):
    return my_string.strip().replace(" ", substitute)


def get_filename(file_path, remove_ext=False):
    '''Extract filename from file path.'''
    _, filename = os.path.split(file_path)
    if remove_ext:
        return filename.split(".")[0]
    else:
        return filename


def delete_folder(d):
    '''
    TO DO: assert that this is a folder
    '''
    if os.path.exists(d):
        shutil.rmtree(d)


def find_matching_file(dir, file_regex):
    pass


def split_list(xs, split_size=None, split_prop=None, random_state=1234):
    '''
    TO DO: ensure valid 
    '''
    assert sum((s != None for s in (split_size, split_prop))
               ) == 1, "Must have exactly one of sample_size or split_size."
    pool = set(xs)
    if split_size:
        left_n = split_size
    else:
        left_n = int(len(xs) * split_prop)

    random.seed(random_state)
    try:
        left = set(random.sample(pool, left_n))
    except ValueError as e:
        print(e)
        left = pool

    right = pool - left

    return left, right


def listdir_full(d):
    return [os.path.join(d, sub) for sub in os.listdir(d)]
