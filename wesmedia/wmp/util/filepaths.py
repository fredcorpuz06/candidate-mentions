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


def get_folder(folder_path):
    '''Extract foldername from file path.

    TO DO: make robust to folder paths w/ & w/o files +
    make sure it uses the ENTIRE file path

    Examples
    --------
    >>> get_folder("candidate-mentions/data/google-images")
    "google-images"
    >>> get_folder("../data/google-images/cand_images.xlsx")
    "google-images"
    >>> get_folder("../random_file")
    ""
    '''
    strs = re.split("[^A-Za-z-_]", folder_path)
    strs = list(filter(lambda x: x != "", strs))
    return strs[-1]


def get_filename(file_path):
    '''Extract filename from file path.'''
    # strs = re.split("[//.]", file_path)
    # strs = list(filter(lambda x: x != "", strs))
    # return strs[-2]
    _, filename = os.path.split(file_path)
    return filename.split(".")[0]


def delete_folder(d):
    '''
    TO DO: assert that this is a folder
    '''
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


def sample_files(
        input_folder, output_folder, type_restriction,
        nimages=10, seed=1234):
    '''Take random sample of images from labelled directory of images.

    TO DO: generalize sampling to list of file types
    '''
    persons = glob.glob(f"{input_folder}/*/")

    all_images = []
    for p in persons:
        img_fps = f"{p}/*.jpg"
        for img_fp in glob.glob(img_fps):
            all_images.append(img_fp)

    if os.path.exists(output_folder):
        delete_folder(output_folder)
    os.makedirs(output_folder)

    random.seed(seed)
    sample_imgs = random.sample(all_images, nimages)
    for img in sample_imgs:
        _, name = os.path.split(img)
        output_file = f"{output_folder}/{name}"
        shutil.copyfile(img, output_file)
