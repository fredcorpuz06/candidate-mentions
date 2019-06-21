import json
import pandas as pd
import glob
from sklearn import linear_model
import sklearn.metrics as mt
import sklearn.model_selection as ms
from joblib import dump, load


def encoding_to_df(json_fp):
    '''Read JSON encoding file to Pandas df.'''
    with open(json_fp) as f:
        face_encodings = json.load(f)

    df = pd.DataFrame.from_dict(face_encodings)
    col_names = [f"ff_{x:03}" for x in range(128)]

    df[col_names] = pd.DataFrame(df.encoding.values.tolist(), index=df.index)
    df = df.drop(columns=['encoding'])

    return df


def create_model():
    # Persons of Interest
    persons = "../data/candidate-images/google-image-query/*.json"
    df = pd.DataFrame([])

    for p in glob.glob(persons):
        df2 = encoding_to_df(p)
        df = df.append(df2, ignore_index=True)

    train, test = ms.train_test_split(df, test_size=0.2, random_state=1234)
    img_meta = ['name', 'img_fp']
    X_train = train.drop(columns=img_meta)
    y_train = train['name']
    X_test = test.drop(columns=img_meta)
    y_test = test['name']

    assert (X_train.shape[0] == y_train.shape[0]) & (
        X_test.shape[0] == y_test.shape[0]), "No. of observations in X & y differ"
    assert (X_train.shape[1] == 128) & (
        X_test.shape[1] == 128), "Face encodings do not have shape of 128"

    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    rez = pd.DataFrame({
        "img_loc": test.img_fp,
        "y_test": test.name,
        "y_pred": y_pred,
        "correct": [x == y for (x, y) in zip(test.name, y_pred)]
    })

    rez.to_csv("../output/poi_pred.csv", index=False)
    print(mt.classification_report(y_test, y_pred))

    dump(clf, "../models/poi_face_clf.joblib")


def main():
    create_model()


if __name__ == "__main__":
    main()
