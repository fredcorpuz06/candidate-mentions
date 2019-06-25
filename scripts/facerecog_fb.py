'''Detect references to a known face in images & videos.

'''
from wmp.analyze import faces
import argparse
import joblib
import pandas as pd
import json
import cv2


def handle_args():
    parser = argparse.ArgumentParser(description="Find candidate references.")
    parser.add_argument()

    args = parser.parse_args()

    return args


def find_mentions(media_folder, clf_loc, out_csv=None):

    # Each found face in an image is a row
    face_dict = faces.encode_unknowns(media_folder, f"{out_csv[:-4]}.json")
    with open(f"{out_csv[:-4]}.json", "r") as fp:
        face_dict = json.load(fp)

    face_df = faces.encoding_df(face_dict)  # img_fp, ff_*  **, ...
    X_new = face_df.drop(columns=["img_fp"])
    clf = joblib.load(clf_loc)

    y_pred = clf.predict(X_new)
    rez = pd.DataFrame({
        "img_fp": face_df.img_fp,
        "y_pred": y_pred,
    })

    rez.to_csv(out_csv, index=False)


def vid_mentions(vid_fp, clf_loc, down_factor=5):
    input_movie = cv2.VideoCapture(vid_fp)

    frame_number = 0
    found_faces = []
    while True:
        ret, frame = input_movie.read()
        frame_number += 1

        if not ret:
            break
        elif frame_number % down_factor != 0:
            continue
        print(frame_number)
        found_faces += faces.encode_face(
            frame, multiple=True, frame_input=True)

    input_movie.release()
    cv2.destroyAllWindows()

    print(f"Found {len(found_faces)} faces")
    input_locs = [f"{vid_fp}_{i:03}" for i in range(len(found_faces))]

    vid_dict = {
        "img_fp": input_locs,
        "encoding": found_faces
    }

    vid_df = faces.encoding_df(vid_dict)
    X_new = vid_df.drop(columns=["img_fp"])

    clf = joblib.load(clf_loc)

    y_pred = clf.predict(X_new)
    rez = pd.DataFrame({
        "img_fp": vid_df.img_fp,
        "y_pred": y_pred,
    })

    rez.to_csv(f"{vid_fp[:-4]}.csv", index=False)


def main():

    MEDIA_FOLDER = "../data/og-ad-media"
    MEDIA_OUT = "../output/og_medias.csv"
    FACE_CLASSIFIER = "../models/poi_face_clf2.joblib"
    # Image classifier
    # mentions_df = find_mentions(
    #     MEDIA_FOLDER, FACE_CLASSIFIER, out_csv=MEDIA_OUT)

    # Video classifier
    SAMPLE_VID = "../data/young_man.mp4"
    rez = vid_mentions(SAMPLE_VID, FACE_CLASSIFIER, down_factor=10)


if __name__ == "__main__":
    main()
