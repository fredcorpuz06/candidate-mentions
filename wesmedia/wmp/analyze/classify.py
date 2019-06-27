'''Create and utilize `sklearn` classifiers on data.
'''

import pandas as pd
import joblib

from . import faces


def classify_batch(face_dict, clf_loc, out_csv=None):
    '''Name faces in new batch of images/frames using pre-trained classifier.

    Parameters
    ----------
    face_dict: dict
        Dictionary with `img_fp` and `encoding` as keys
    clf_loc: str
        File location of `sklearn` classifier saved in joblib file
    out_csv: str
        Output file location of CSV

    Returns
    -------
    Pandas dataframe with `img_fp` and `y_pred` columns
    '''
    face_df = faces.encoding_df(face_dict)  # img_fp, ff_*  **, ...
    X_new = face_df.drop(columns=["img_fp"])
    if X_new.shape == (0, 128):
        raise PredictorError("There are 0 rows in this batch")

    clf = joblib.load(clf_loc)

    y_pred = clf.predict(X_new)
    rez = pd.DataFrame({
        "img_fp": face_df.img_fp,
        "y_pred": y_pred,
    })

    if out_csv:
        rez.to_csv(out_csv, index=False)

    return rez


class Error(Exception):
    '''Base class for other errors in this module.'''
    pass


class PredictorError(Error):
    '''Exception raised for invalid input into classifier.'''

    def __init__(self, msg):
        self.msg = msg
