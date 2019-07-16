import os
import dlib
import numpy as np
from PIL import Image
from collections import namedtuple
detector = dlib.get_frontal_face_detector()
# Trained facial shape predictor and recognition model from:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
FACE_LANDMARKS_68 = "../models/shape_predictor_68_face_landmarks.dat"
# http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat
FACE_RECOGNITION_MODEL = "../models/dlib_face_recognition_resnet_model_v1.dat"

sp = dlib.shape_predictor(FACE_LANDMARKS_68)
facerec = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)


Box = namedtuple("Box", ["left", "top", "right", "bottom"])


def _raw_face_rects(img, number_of_times_to_upsample=1, model="hog"):
    """
        array of bounding boxes of human faces in a image
        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                    deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of dlib 'rect' objects of found face locations
        """
    return detector(img, number_of_times_to_upsample)


def _raw_face_landmarks(face_image, face_rects):
    '''Get the landmarks/parts for the face in box d.

    Ask the detector to find the bounding boxes of each face. The 1 in the
    second argument indicates that we should upsample the image 1 time. This
    will make everything bigger and allow us to detect more faces.

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations

    '''
    return [sp(face_image, face_rect) for face_rect in face_rects]


def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def _to_box(r, img_shape=None):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param img_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    if img_shape:
        face_box = Box(
            max(r.left(), 0),
            max(r.top(), 0),
            min(r.right(), img_shape[1]),
            min(r.bottom(), img_shape[0]))
    else:
        face_box = Box(r.left(), r.top(), r.right(), r.bottom())
    return face_box


def _raw_face_locations(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image
    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    return [_to_box(face_rect, img.shape) for face_rect in _raw_face_rects(img, number_of_times_to_upsample)]


def _raw_face_encodings(
        face_image, return_boxes=False,
        number_of_times_to_upsample=1, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.
    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    face_rects = _raw_face_rects(face_image, number_of_times_to_upsample)
    raw_landmarks = _raw_face_landmarks(face_image, face_rects)
    face_encodings = [np.array(facerec.compute_face_descriptor(
        face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

    if return_boxes:
        face_boxes = [_to_box(face_rect, face_image.shape)
                      for face_rect in face_rects]
        return face_boxes, face_encodings
    else:
        return face_encodings


def encode_image(face_image, number_of_times_to_upsample=1, num_jitters=1):
    boxes, encodings = _raw_face_encodings(
        face_image, return_boxes=True)
    new_faces = []
    for b, e in zip(boxes, encodings):
        new_face = Face(b, e, face_image)
        new_faces.append(new_face)
    return new_faces


class Face:
    _PAD_PERC = 0.35

    def __init__(self, box, encoding, full_image):
        self.box = Box._make(box)
        self.encoding = encoding
        self._full_image_data = full_image
        self.image_data = self._crop_face(full_image, pad_perc=self._PAD_PERC)
        self.source_filename = None
        self.name = None

    def __repr__(self):
        return (f"Face(image_source={self.source_filename}, "
                f"box={self.box}, name={self.name})")

    def _crop_face(self, image, pad_perc=None):
        '''Crops image to a box around found face w/ optional padding.'''
        if pad_perc:
            l, w, *_ = image.shape
            l_pad = int((self.box.bottom - self.box.top) * pad_perc)
            w_pad = int((self.box.right - self.box.left) * pad_perc)
            top_pad = max(self.box.top - l_pad, 0)
            bottom_pad = min(self.box.bottom + l_pad, l)
            left_pad = max(self.box.left - w_pad, 0)
            right_pad = min(self.box.right + w_pad, w)

            return image[top_pad:bottom_pad, left_pad:right_pad]
        else:
            return image[self.box.top:self.box.bottom,
                         self.box.left:self.box.right]

    def tag_source(self, file_loc, frame_num=None):
        '''Mark file source of found face.'''
        file_name = filepaths.get_filename(file_loc)
        if frame_num is not None:
            return f"Vid: {file_name}; Frame: {frame_num}"
        else:
            return f"Img: {file_name}"

    def compare_to(self, other_face, threshold=0.6):
        '''Determines if the two encoded faces are of the same person.

        Uses Euclidian distance of the two 128d face encodings and a threshold
        value to determine similarity.
        '''
        distance = np.linalg.norm(self.encoding - other_face.encoding)
        return distance < threshold

    def write_face(self, output_loc):
        '''Writes image of cropped face to output location'''
        try:
            im = Image.fromarray(self.image_data)
            im.save(output_loc)
            print(f"Wrote cropped face to {output_loc}")
        except:
            print(f"--- Unknown error at {output_loc}")


class FaceBatch:
    def __init__(self, faces):
        self.batch_faces = faces
        self.encoding_names, self.encoding_matrix = self._create_matrix()

    def __len__(self):
        return len(self.batch_faces)

    def __getitem__(self, idx):
        return self.batch_faces[idx]

    def __repr__(self):
        return (f"FaceBatch(len(batch_faces)={self.__len__()}, "
                f"encoding_matrix.shape={self.encoding_matrix.shape})")

    def _create_matrix(self):
        names, encodings = [], []
        for f in self.batch_faces:
            names.append(f.name)
            encodings.append(f.encoding)
        return names, np.stack(encodings, axis=0)

    def compare_batch(self, other):
        '''
        Parameters
        ----------
        self.encoding_matrix: np.array
            Array of face encodings of unknown people
        other.encoding_matrix: np.array
            Array of face encodings of known people
        threshold: float

        Returns
        -------
        np.array of bools
        '''

        assert self.encoding_matrix.shape[1] == other.encoding_matrix.shape[1]
        n_unknown, n_known = self.encoding_matrix.shape[0], other.encoding_matrix.shape[0]
        print(
            f"We are comparing {n_unknown} unknown faces "
            f"to {n_known} known (reference) faces")

        x0 = np.repeat(
            self.encoding_matrix[:, np.newaxis, :], n_known, axis=1)
        x1 = np.repeat(
            other.encoding_matrix[np.newaxis, :, :], n_unknown, axis=0)
        distance = np.linalg.norm(x0 - x1,  axis=2)
        print(f"Output dimensions of bool array is {distance.shape}")
        return distance

    def apply_comparison(self, names, comp_matrix, root_dir, threshold=0.6):
        def decide_name(xs):
            '''Identify single name for each `face` that meets threshold similarity.'''
            name_idxs, *_ = np.where(xs < threshold)

            if len(name_idxs) == 0:
                return "Unknown"
            elif len(name_idxs) == 1:
                return names[name_idxs[0]]
            else:
                # Multiple people match below threshold
                return names[np.argmin(xs)]

        face_names = [decide_name(comp1) for comp1 in comp_matrix]

        # Write face to correct folder
        for i, name in enumerate(face_names):
            self.batch_faces[i].name = name
            filename = self.batch_faces[i].source_filename.split(".")[0]
            l, t = self.batch_faces[i].box[:2]
            filename = f"{filename}_{l}_{t}.jpg"
            filepath = os.path.join(root_dir, name, filename)
            self.batch_faces[i].write_face(filepath)


def main():
    IMG_FILE = "../data/google-images/Barack_Obama_raw/Barack_Obama_014.jpg"
    img = load_image_file(IMG_FILE)
    locations = _raw_face_locations(img)
    print(locations)
    encodings = _raw_face_encodings(img)
    print(len(encodings)[0])


if __name__ == "__main__":
    main()
