'''Encode faces using `face_recognition`. 
'''

import face_recognition
import cv2
import json

import re
from collections import Counter
import glob
import os
import shutil
import random
import pandas as pd


def encode_face(img_loc):
    '''Encode 1 face in an input image.

    Parameters
    ----------
    img_loc: str
        File path of image.

    Returns
    -------
    list 
        Face encoding in a 128-length float list.

    Examples
    --------
    >>> encode_face("trump_000.jpg")
    [-0.17839303612709045, 0.2754783630371094, ..., 0.022344175726175308]

    '''
    image = cv2.imread(img_loc)
    if image is None:
        raise ValueError("Input image cannot be read by OpenCV.")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # dlib ordering (RGB)
    boxes = face_recognition.face_locations(
        rgb, model="hog")  # cnn or hog

    encodings = face_recognition.face_encodings(rgb, boxes)
    if len(encodings) == 0:
        raise FacesError("Input image contains no faces")
    elif len(encodings) > 1:
        raise FacesError("Input image contains multiple faces")

    face_encoding = encodings[0].tolist()

    return face_encoding


def get_last_str(folder_path):
    '''Extract rightmost complete string.'''
    strs = re.split("[^A-Za-z-_]", folder_path)
    strs = list(filter(lambda x: x != "", strs))
    return strs[-1]


def encode_person(image_folder):
    '''Encode all images of single person in a folder.

    Parameters:
    -----------
    folder_path: str
        Folder path

    Returns
    -------
    name: str
        Name of person in image folder.
    encodings: list, shape = [n_images, ]
        The list of 128-D face encodings.

    Examples
    --------
    >>> encode_person("google-images/blclinton")
    [
        [-0.17839303612709045, 0.2754783630371094, ...],
        [-0.12661609045393930, 0.05347478363271094, ...],
        ...
    ]

    '''
    img_fps = f"{image_folder}/*.jpg"
    name = get_last_str(image_folder)

    names, valid_imgs, encodings = [], [], []
    for img_fp in glob.glob(img_fps):
        try:
            encoding = encode_face(img_fp)
        except (ValueError, FacesError) as e:
            print(f"{img_fp} raised {e}")
        else:
            encodings.append(encoding)
            _, img_name = os.path.split(img_fp)
            valid_imgs.append(img_name)
            names.append(name)

    return {
        "name": names,
        "img_fp": valid_imgs,
        "encoding": encodings
    }


def delete_folder(folder_path):
    shutil.rmtree(folder_path)


def sample_images(input_folder, output_folder, nimages=10, seed=1234):
    '''Take random sample of images from labelled directory of images.'''
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


def save_encodings(person_encodings, out_fp):
    '''Save names and encodings to a JSON file.

    Parameters
    ----------
    names: list of str
        Ordered list of names.
    img_fps: list of str
        Ordered list of file names of images.
    encodings: list of list
        Ordered list of 128-D face encodings.
    out_fp: str
        Output file name.

    '''
    assert len(person_encodings['name']) == len(
        person_encodings['encoding']), "Names does not match encodings"
    assert len(person_encodings['img_fp']) == len(
        person_encodings['encoding']), "Image files does not match encodings"

    with open(out_fp, "w") as fp:
        json.dump(person_encodings, fp, indent=4, sort_keys=True)


class Error(Exception):
    '''Base class for other errors in this module.'''
    pass


class FacesError(Error):
    '''Exception raised for wrong number of faces in the input.'''

    def __init__(self, msg):
        self.msg = msg


class FaceRecognizer:
    '''FaceRecognizer contains functions to aid in identifying faces in images.

    Attributes:
        known_names (list): Strings to identify the face_encodings in 'known_faces'
        known_faces (list): 128-D encodings of faces from 'face_recognition' Python package

    '''

    def __init__(self, known_names, known_faces):
        self.known_names = known_names
        self.known_faces = known_faces

    def read_image_rgb(self, face_fp):
        image = cv2.imread(face_fp)
        if image is None:
            return None
        return self.convert_image_rgb(image)

    def convert_image_rgb(self, bgr_img):
        '''Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses).
        '''
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    def find_faces(self, rgb_img, model='hog'):
        '''
        Args:
            rgb_img: Image data readable by face_recognition in RGB format
            model: Method to be used to encode faces. Defaults to "hog". 
                Options are "hog" and "cnn"
        '''
        face_locations = face_recognition.face_locations(rgb_img, model=model)
        face_encodings = face_recognition.face_encodings(
            rgb_img, face_locations)
        return {
            "face_locations": face_locations,
            "face_encodings": face_encodings,
        }

    def match_unknowns(self, unknown_faces, tolerance=0.6):
        '''Finds closest match in known_faces for each unknown face.

        Find name of 1 person who had the most knownEncodings
        that match the unknown face. If no match is found, name
        the face "Unknown".

        Args:
            face_pairings: Dict containing "face_locations" and 
                "face_encodings"
            tolerance: threshold for face distance calculated to
            qualify as a match; lower numbers make face comparisons 
            more strict

        Returns:
            A dict containing the input "face_locations" and 
            "face_encodings" with the found "face_names".
        '''
        face_names = []  # name per person found
        for face_encoding in unknown_faces["face_encodings"]:
            match = face_recognition.compare_faces(
                self.known_faces, face_encoding, tolerance=tolerance)
            name = "Unknown"

            if True in match:
                matchedIdxs = [i for (i, b) in enumerate(match) if b]
                counts = {}

                for i in matchedIdxs:
                    name = self.known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            face_names.append(name)

        unknown_faces['face_names'] = face_names

        return unknown_faces

    def draw_matches(self, rgb_img, known_faces):
        '''Draw box and label around each found face in input image'''
        loc_faces = zip(known_faces['face_locations'],
                        known_faces['face_names'])
        for (top, right, bottom, left), name in loc_faces:
            # print((top, right, bottom, left), name)
            cv2.rectangle(rgb_img, (left, top), (right, bottom),
                          (0, 255, 0), 2)  # box
            y = top - 15 if top - 15 > 15 else top + 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_img, name, (left, y),
                        font, 0.75, (0, 255, 0), 2)  # label name below face
        # return rgb_img

    def name_all_faces(self, rgb_img, draw_matches=False):
        '''Find all faces in an image.

        Args:
            rgb_img: Numpy array representing an image
            draw_matches: Bool to indicate whether function returns an image object
                with the faces found labelled with boxes + names

        Returns:
            A dict with all info about the faces found. Here is an example with 2
            found faces:
            {
                "face_locations": [(64, 408, 219, 253), (410, 229, 33, 41)],
                "face_encodings": [array(), array()],
                "face_names": ["JOSH_HAWLEY", "FRED_MANCHIN"]
                "boxed_img" (numpy.ndarray rep. an image) 
            }


        '''
        unknown_faces = self.find_faces(rgb_img)
        found_faces = self.match_unknowns(unknown_faces)
        found_faces['boxed_img'] = rgb_img
        if draw_matches:
            self.draw_matches(rgb_img, found_faces)
            found_faces['boxed_img'] = rgb_img

        return found_faces


class VideoAnnotator(FaceRecognizer):

    def __init__(self, known_names, known_faces, down_factor=None):
        super().__init__(known_names, known_faces)
        if down_factor == None:
            self.down_factor = 5
        self.down_factor = down_factor  # only process every Nth frame

    def get_vid_specs(self, input_movie):
        fps = int(input_movie.get(cv2.CAP_PROP_FPS))  # input: FPS (usually 30)
        width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

        return (fps, (width, height), length)

    def process_vid(self, vid_fp, out_vid_fp):
        input_movie = cv2.VideoCapture(vid_fp)

        fps, dim, length = self.get_vid_specs(input_movie)
        out_fps = fps / self.down_factor
        # Create an output movie file (make sure resolution/frame rate matches input video!)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter(out_vid_fp,
                                       fourcc, out_fps, dim)

        all_found_faces = []
        frame_number = 0
        while True:
            ret, frame = input_movie.read()
            frame_number += 1

            if not ret:
                break
            elif frame_number % self.down_factor != 0:
                continue

            found_faces = self.name_all_faces(frame, draw_matches=True)
            all_found_faces += found_faces['face_names']

            print(f"Writing frame {frame_number} / {length}")
            output_movie.write(found_faces['boxed_img'])

        input_movie.release()
        cv2.destroyAllWindows()

        c = Counter(all_found_faces)  # all names mentioned in more than 2 sec
        print(c.items())
        return [name for name, n in c.items() if n > out_fps]
