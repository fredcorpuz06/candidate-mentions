import json
import pandas as pd
import glob
from sklearn import linear_model
import sklearn.metrics as mt
import sklearn.model_selection as ms
from joblib import dump, load
from wmp.analyze import faces


def create_model():
    # Persons of Interest
    persons = "../data/candidate-images/google-image-query/*.json"
    df = pd.DataFrame([])

    for p in glob.glob(persons):
        df2 = faces.encoding_df(json_fp=p)
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

    rez.to_csv("../output/poi_pred2.csv", index=False)
    print(mt.classification_report(y_test, y_pred))

    dump(clf, "../models/poi_face_clf2.joblib")


def main():
    create_model()


if __name__ == "__main__":
    main()
