'''Download txt file of links.

Code adapted from
https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

Usage:
    python pull_google_images -u clinton.txt
'''
# import the necessary packages
from imutils import paths
import argparse
import requests
import cv2
import os

ROOT_FOLDER = "../data/candidate-images/google-image-query"

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
                help="path to file containing image URLs")
args = vars(ap.parse_args())

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
urls_loc = f"{ROOT_FOLDER}/{args['urls']}"
person_name = args['urls'].split(".")[0]

output_loc = f"{ROOT_FOLDER}/{person_name}"

if not os.path.exists(output_loc):
    os.makedirs(output_loc)


rows = open(urls_loc).read().strip().split("\n")
total = 0

# loop the URLs
for url in rows:
    try:
        # try to download the image
        r = requests.get(url, timeout=60)

        # save the image to disk
        p = f"{output_loc}/{person_name}_{str(total).zfill(3)}.jpg"
        # p = os.path.sep.join([output_loc, "{}_{}.jpg".format(
        #     person_name, str(total).zfill(8))])
        with open(p, "wb") as f:
            f.write(r.content)

        # update the counter
        print("[INFO] downloaded: {}".format(p))
        total += 1

    # handle if any exceptions are thrown during the download process
    except:
        print("[INFO] error downloading {}...skipping".format(p))

# loop over the image paths we just downloaded
for imagePath in paths.list_images(output_loc):
    # initialize if the image should be deleted or not
    delete = False

    # try to load the image
    try:
        image = cv2.imread(imagePath)

        # if the image is `None` then we could not properly load it
        # from disk, so delete it
        if image is None:
            delete = True

    # if OpenCV cannot load the image then the image is likely
    # corrupt so we should delete it
    except:
        print("Except")
        delete = True

    # check to see if the image should be deleted
    if delete:
        print("[INFO] deleting {}".format(imagePath))
        os.remove(imagePath)
