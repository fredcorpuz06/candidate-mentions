from wmp.collect.webpage import GoogleImages
from wmp.analyze import faces
from wmp.util import filepaths
import glob
import os
import re


def find_reference_face(query_folder, reference_tag=None):
    def single_face_encoding(image):
        potential_faces = faces.encode_image(image)
        if len(potential_faces) == 1:
            print(f"Reference image: {image}")
            return potential_faces[0]
        else:
            return None

    if reference_tag:
        image = [image for image in os.listdir(query_folder) if re.match(
            reference_tag, image) is not None][0]
        image = os.path.join(query_folder, image)

    reference_face = single_face_encoding(image)
    if reference_face is not None:
        return reference_face
    else:
        # guess first in folder works
        for i in os.listdir(query_folder)[1:]:
            reference_face = single_face_encoding(i)
            if reference_face:
                return reference_face
            else:
                continue

    raise faces.FacesError("No reference face found")


def main():
    WEBDRIVER_FP = "../data/chromedriver_win32/chromedriver.exe"
    # OUT_FOLDER = "../data/google-images"
    OUT_FOLDER = "../data/google-senate18"
    POLITICIANS_LIST = "../data/politicians.txt"
    # POLITICIANS_LIST = "../data/senate_candidates_2018.txt"
    POLITICIANS_LIST = "../data/re_do_facecrop.txt"

    # queries = ["Chris_Murphy"]

    # Download images into own directory
    with open(POLITICIANS_LIST, "r") as f:
        queries = f.readlines()
        queries = [q.rstrip("\n") for q in queries]

    # gi = GoogleImages(WEBDRIVER_FP, OUT_FOLDER)

    # TO DO: multiprocessing each query + face cropping
    for q in queries:
        print(f"\nGoogle Query: {q}\n")
        # query_folder = gi.image_search(q, nimages=400)  # Kamala_Harris_raw
        query_folder = f"{OUT_FOLDER}/{q}_raw"
        ref_face = find_reference_face(query_folder, reference_tag="reference")
        trainset_folder = filepaths.replace_spaces(q, substitute="_")
        trainset_folder = f"{OUT_FOLDER}/{trainset_folder}"
        filepaths.create_folder(trainset_folder)
        _ = faces.create_trainset(
            query_folder, trainset_folder, ref_face)  # Kamala_Harris


if __name__ == "__main__":
    main()
