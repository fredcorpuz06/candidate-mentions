'''Move downloaded images into train/valid folders to train Pytorch model.
'''
from wmp.util import filepaths as fp
import shutil
import os
import re


def locate_wanted(root_dir):
    def not_raw(my_str):
        return re.match(r"[A-Za-z_]+_raw", my_str) is None

    all_folders = os.listdir(root_dir)
    select_folders = list(filter(not_raw, all_folders))

    return [os.path.join(root_dir, s) for s in select_folders]


def main():

    # ROOT_FOLDER = "../data/google-senate18"
    ROOT_FOLDER = "../data/google-images"
    DATASET_FOLDER = "../data/faces-dataset-senate18"
    # Locate folders of images to move (not raw)
    wanted_folders = locate_wanted(ROOT_FOLDER)

    train_folder = f"{DATASET_FOLDER}/train"
    valid_folder = f"{DATASET_FOLDER}/valid"
    sample_train = f"{DATASET_FOLDER}/sample/train"
    sample_valid = f"{DATASET_FOLDER}/sample/valid"
    fp.create_folder(train_folder, valid_folder,
                     sample_train, sample_valid)

    for src in wanted_folders[-2:]:
        person_name = fp.get_folder(src)
        dst_train = os.path.join(train_folder, person_name)
        dst_valid = os.path.join(valid_folder, person_name)
        dst_sample_t = os.path.join(sample_train, person_name)
        dst_sample_v = os.path.join(sample_valid, person_name)
        fp.create_folder(dst_train, dst_valid, dst_sample_t, dst_sample_v)

        # Copy 80/20 % of images to train/valid
        left, right = fp.split_list(
            os.listdir(src), split_prop=0.8, random_state=1234)

        for i in left:
            shutil.copy2(os.path.join(src, i), os.path.join(dst_train, i))
        for i in right:
            shutil.copy2(os.path.join(src, i), os.path.join(dst_valid, i))
        print((f"{person_name} has {len(left)} images in train "
               f"and {len(right)} in valid"))

        # Take sample of 40/10 images for each person
        sample, _ = fp.split_list(
            os.listdir(src), split_size=50, random_state=1234)
        left, right = fp.split_list(sample, split_prop=0.8, random_state=1234)
        for i in left:
            shutil.copy2(os.path.join(src, i), os.path.join(dst_sample_t, i))
        for i in right:
            shutil.copy2(os.path.join(src, i), os.path.join(dst_sample_v, i))

        print((f"{person_name} has {len(left)} images in sample train "
               f"and {len(right)} in sample valid"))


if __name__ == "__main__":
    main()
