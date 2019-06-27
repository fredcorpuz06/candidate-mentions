'''Detect references to a known face in images & videos.
'''
from wmp.analyze import faces, classify
from wmp.analyze.classify import PredictorError
import argparse
import joblib
import pandas as pd
import json
import cv2
import glob
import numpy as np


def image_mentions(media_folder, clf_loc, out_csv=None):

    # face_dict = faces.encode_images(media_folder, f"{out_csv[:-4]}.json")
    face_dict = faces.encode_images(media_folder)
    _ = classify.classify_batch(face_dict, clf_loc, out_csv)
    predicted_faces_df = classify.classify_batch(face_dict, clf_loc)
    return predicted_faces_df


def vid_mentions(vid_fp, clf_loc, down_factor=5):
    vid_dict = faces.encode_video(vid_fp)
    # out_name = f"{vid_fp[:-4]}.csv"
    # _ = classify.classify_batch(vid_dict, clf_loc, out_csv=out_name)
    try:
        predicted_faces_df = classify.classify_batch(vid_dict, clf_loc)
    except PredictorError as e:
        print(f"--- {vid_fp} skipped. Raised {e}")
        predicted_faces_df = pd.DataFrame({
            "img_fp": [faces.get_filename(vid_fp)],
            "y_pred": [np.nan]
        })

    return predicted_faces_df


def multimedia_mentions(media_folder, clf_loc, out_csv=None):
    print(f"Processing images in {media_folder}")
    images_predicted = image_mentions(media_folder, clf_loc, out_csv=out_csv)

    videos_predicted = pd.DataFrame([])
    vid_fps = f"{media_folder}/*.mp4"

    for vid_fp in glob.glob(vid_fps):
        print(f"Processing {vid_fp}")
        vid_pred = vid_mentions(vid_fp, clf_loc, down_factor=30)
        vid_pred_summary = pd.DataFrame({
            "img_fp": vid_pred.img_fp[0],
            "y_pred": [vid_pred.y_pred.unique().tolist()]
        })
        videos_predicted = videos_predicted.append(
            vid_pred_summary, ignore_index=True)

    multi_predicted = images_predicted.append(
        videos_predicted, ignore_index=True)
    if out_csv:
        multi_predicted.to_csv(out_csv, index=False)

    return multi_predicted


def main():

    MEDIA_FOLDER = "../data/og-ad-media"
    MEDIA_OUT = "../output/og_medias.csv"
    FACE_CLASSIFIER = "../models/poi_face_clf2.joblib"
    # Image classifier
    # _ = image_mentions(MEDIA_FOLDER, FACE_CLASSIFIER, out_csv=MEDIA_OUT)

    # Video classifier
    SAMPLE_VID = "../data/sample-vid-reference/trump_still.mp4"
    # _ = vid_mentions(SAMPLE_VID, FACE_CLASSIFIER, down_factor=10)

    # Multimedia classifier
    _ = multimedia_mentions(MEDIA_FOLDER, FACE_CLASSIFIER, out_csv=MEDIA_OUT)
    # print(faces.get_filename("../data/sample-vid-reference/trump_still.mp4"))


if __name__ == "__main__":
    main()
