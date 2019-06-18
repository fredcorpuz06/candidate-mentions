'''Determine sentiment of text (promotional or attack ad).'''
import unittest
from enum import Enum

import numpy as np 
import pandas as pd 
import sklearn.linear_model as lm 
import sklearn.model_selection as ms
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as mt
from joblib import dump, load

class Sentiment(Enum):
    ATTACK = -1
    PROMOTE = 1

def get_sentiment(text, model_loc, vectorizer_loc):
    vectorizer = load(vectorizer_loc)
    clf = load(model_loc)

    X_test = vectorizer.transform([text])
    y_pred = clf.predict(X_test)[0]
    
    if y_pred == "promote":
        return Sentiment.PROMOTE
    else:
        return Sentiment.ATTACK

def create_model():
    df = pd.read_csv("../data/tv-ads-kantar-meta/tv_ads_transcribed_clean.csv")
    
    train, test = ms.train_test_split(df, test_size=0.2, random_state=1234)
    
    vectorizer = CountVectorizer(stop_words='english', max_df=0.98, min_df=0.05)
    X_train = vectorizer.fit_transform(train['transcript'])
    y_train = train['tone']
    # print(X_train.shape, y_train.shape)

    X_test = vectorizer.transform(test['transcript'])
    y_test = test['tone']

    log_reg = lm.LogisticRegression(solver='newton-cg')
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    print(mt.classification_report(y_test, y_pred))

    dump(vectorizer, "../models/vectorizer_sample.joblib")
    dump(log_reg, "../models/log_reg_sample.joblib")


class TestSentiment(unittest.TestCase):

    promote = '''hard work and honest living at the American promise a good
                 life in the middle class today big government and big business
                 could take that life away I'm Josh Hawley I've never been part
                 of the old battles in Washington I don't know anything anyone
                 has people fight back truth is we're running out of time to
                 save our country and that's why I approve this message'''

    attack = '''I support forcing insurance companies to cover all
                pre-existing conditions The Golden Boy Josh Hawley is lying to
                us again Ollie's ad has been called ridiculously dishonest
                cleaning to support pre-existing conditions while backing a
                lawsuit to end them it's Holly who filed a lawsuit to overturn
                protections for people with pre-existing conditions like cancer
                and diabetes Josh Hawley he's lying right to our faces SMP is 
                responsible for the content of the'''
    promote2 = '''someone is sexually assaulted in the United States
                  protecting our families means putting sexual predators
                  Behind Bars attorney general Josh Hawley uncovered thousands
                  of untested rape kits sitting in storage DNA evidence that
                  can now be used to help bring sexual predators to Justice
                  Josh Hawley is fighting for solutions that get Justice for
                  victims and strengthen law enforcement protect your family
                  Birds Josh Hawley to keep working to get sexual predators
                  off our streets'''

    def test_get_sentiment(self):
        self.assertEqual(
            get_sentiment(
                self.promote,
                "../models/log_reg_sample.joblib",
                "../models/vectorizer_sample.joblib"), Sentiment.PROMOTE)
        self.assertEqual(
            get_sentiment(
                self.attack,
                "../models/log_reg_sample.joblib",
                "../models/vectorizer_sample.joblib"), Sentiment.ATTACK)
        self.assertEqual(
            get_sentiment(
                self.promote2,
                "../models/log_reg_sample.joblib",
                "../models/vectorizer_sample.joblib"), Sentiment.PROMOTE)

if __name__ == "__main__":
    unittest.main()