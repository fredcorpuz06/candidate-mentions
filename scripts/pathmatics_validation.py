'''Visualize results of `face_recogntion` OOB
'''
from wmp.analyze import faces2 as faces
from wmp.util import filepaths
import numpy as np
import os
import shutil


def move_reference_images(in_dir, out_dir):
    '''Grab 2nd image in a directory and copy to own folder'''
    for person_folder in fp.listdir_full(in_dir):
        # grab first image
        s = fp.listdir_full(person_folder)[0]
        # copy + rename
        _, ref_name = os.path.split(person_folder)
        out_person_dir = os.path.join(out_dir, ref_name)
        fp.create_folder(out_person_dir)
        d = os.path.join(out_person_dir, "_reference.jpg")
        shutil.copy2(s, d)


def encode_reference_images(ref_img_dir):
    '''Encode each reference image to a `face` object. 

    Parameters
    ----------
    ref_img_dir: file path to directory

    The folder structure of the image directory is as follows:
    ref_img_dir
        ├───Donald_Trump
        │       _reference.jpg
        │
        ├───Hillary_Clinton
        │       _reference.jpg
        │
        └───Bernie_Sanders
                _reference.jpg

    Returns
    -------
    list
        list of face objects
    '''
    ref_faces = []
    for person in os.listdir(ref_img_dir):
        img_fp = os.path.join(ref_img_dir, person, "_reference.jpg")
        img_data = faces.load_image_file(img_fp)
        ref_face = faces.encode_image(img_data)[0]
        ref_face.name = person
        ref_face.source_filename = img_fp
        ref_faces.append(ref_face)

    return ref_faces


def main():
    NEW_IMAGES_DIR = "../data/wmp-coded/pathmatics_creatives2018"
    REFERENCE_SOURCE = "../data/faces-dataset-senate18/train"
    OUTPUT_DIR = "../data/pathmatics_facerecog"
    # create folders for each politician: NAME/_reference.jpg
    # move_reference_images(REFERENCE_SOURCE, OUTPUT_DIR)
    filepaths.delete_folder(os.path.join(OUTPUT_DIR, "Others"))
    filepaths.delete_folder(os.path.join(OUTPUT_DIR, "Unknown"))
    reference_faces = encode_reference_images(OUTPUT_DIR)

    # multiprocessing!
    ref_batch = faces.FaceBatch(reference_faces)
    print("Reference faces: ", ref_batch)
    # find all faces in Pathmatics dataset => write to correct folder
    filepaths.create_folder(os.path.join(OUTPUT_DIR, "Unknown"))
    unknown_faces = []
    for screenshot in os.listdir(NEW_IMAGES_DIR):
        fp = os.path.join(NEW_IMAGES_DIR, screenshot)
        img_data = faces.load_image_file(fp)
        fs = faces.encode_image(img_data)
        for f in fs:
            f.source_filename = screenshot
        unknown_faces += fs
    unknown_batch = faces.FaceBatch(unknown_faces)
    print("Unknown faces: ", unknown_batch)
    comp_matrix = unknown_batch.compare_batch(ref_batch)
    unknown_batch.apply_comparison(
        ref_batch.encoding_names, comp_matrix, OUTPUT_DIR)

    # [print(u.name) for u in unknown_batch.batch_faces]


if __name__ == "__main__":
    main()
