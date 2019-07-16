# from imutils import build_montages
import os
import numpy as np
import cv2


def build_montages(image_list, image_shape, montage_shape, image_labels=None):
    """
    ---------------------------------------------------------------------------------------------
    author: Kyle Hounslow
    ---------------------------------------------------------------------------------------------
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format
    ---------------------------------------------------------------------------------------------
    example usage:
    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = make_montages_of_images(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)
    ----------------------------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception(
            'image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception(
            'montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    # start with black canvas to draw images onto
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                             dtype=np.uint8)
    cursor_pos = [0, 0]
    start_new_img = False
    for img, label in zip(image_list, image_labels):
        if type(img).__module__ != np.__name__:
            raise Exception(
                'input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = cv2.resize(img, image_shape)
        img = label_img(img, label)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1],
                      cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                                         dtype=np.uint8)
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages


def grab_images(in_dir):
    ref_images = []
    for person_folder in os.listdir(in_dir):
        if person_folder == "Unknown":
            continue
        else:
            image_fp = os.path.join(in_dir, person_folder, "_reference.jpg")
            ref_images.append((person_folder, image_fp))
    return ref_images


def label_img(img, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2
    cv2.putText(
        img,
        label,
        location,
        font,
        fontScale,
        fontColor,
        lineType)
    return img


def main():
    REFERENCE_SOURCE = "../data/pathmatics_facerecog"
    image_fps = grab_images(REFERENCE_SOURCE)
    labels = [i[0] for i in image_fps]
    images = [cv2.imread(i[1]) for i in image_fps]
    montages = build_montages(images, (256, 256), (5, 5), image_labels=labels)

    # loop over the montages and display each of them
    for idx, montage in enumerate(montages):
        cv2.imwrite(
            f"../data/politician_montages/politician_montage_{idx}.jpg", montage)
        # cv2.imshow("Montage", montage)
        # cv2.waitKey(0)


if __name__ == "__main__":
    main()
