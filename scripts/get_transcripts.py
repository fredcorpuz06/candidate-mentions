import sqlite3
from threading import Thread, Lock
import pandas as pd

import os
import re
import argparse

import google_transcribe

class TVAd:
    '''Each TVAd has a FLAC file in a directory.'''

    def __init__(self, adid, tone):
        self.id = adid
        self.tone = tone
        self.transcript = ''


class DbActions:
    '''Connection and actions to specified sqlite3 database.'''

    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.lock = Lock()

    def insert_ad(self, ad):
        self.lock.acquire()
        with self.conn:
            self.cursor.execute(
                'INSERT INTO tvads VALUES (:id, :tone, :transcript)',
                {'id': ad.id, 'tone': ad.tone, 'transcript': ad.transcript})
        self.lock.release()

    def get_ads_by_tone(self, tone):
        self.cursor.execute(
            'SELECT * FROM tvads WHERE tone = :tone',
            {'tone': tone})
        return self.cursor.fetchall()

    def remove_ad(self, ad):
        self.lock.acquire()
        with self.conn:
            self.cursor.execute(
                'DELETE FROM tvads WHERE id = :id',
                {'id': ad.id})
        self.lock.release()


class FastGoogleTranscribe:
    '''Threaded requests from Google Cloud Speech.'''
    def __init__(self, ads, db_name, media_folder, n):
        self.ads = ads
        self.db_name = db_name
        self.media_folder = media_folder
        self.n = n
        self.start()
        
    def start(self):
        self.threaded_gets()

    def get_transcript(self, ad):
        transcript = google_transcribe.transcribe_file(
            f"{self.media_folder}/{ad.id}.flac")
        
        return transcript

    def get_batch_transcript(self, ads_sub):
        db_act = DbActions(self.db_name) # each has own connection
        for ad in ads_sub:
            ad.transcript = self.get_transcript(ad)
            db_act.insert_ad(ad)
        return db_act

    def threaded_gets(self):
        '''Runs get_batch_transcript in `n` concurrent threads.'''
        threads = []
        n = 8
        for i in range(n):
            ads_sub = self.ads[i::n] # i, i+nthread, i+2*nthread
            t = Thread(target=self.get_batch_transcript, args=(ads_sub,))
            threads.append(t)

        [t.start() for t in threads]
        [t.join() for t in threads] # wait for operations to finish


def df_to_tvads(df):
    '''Converts a Pandas DataFrame to a list of TVAd objects.'''
    rows = df.to_dict('records')
    tvads = []
    for row in rows:
        ad_id = clean_file_name(row['alt'])
        tone = row['ad_tone']
        tvads.append(TVAd(ad_id, tone))
    return tvads

def read_cmd_args(max_rows, ave_threads):
    '''Reads optional arguments from command line'''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--nrows', '-r', default=max_rows, type=int,
        help='sum the integers (default: find the max)')
    parser.add_argument(
        '--nthreads', '-t', default=ave_threads, type=int,
        help='sum the integers (default: find the max)')

    args = parser.parse_args()

    return args.nrows, args.nthreads

def clean_file_name(file_name):
    '''Ensures validity of file name.'''
    return re.sub("[&?=,']", "_", file_name) 

def main():
    DF_LOC = '../data/tv-ads-kantar-meta/tv_ads_attack.csv'
    DB_NAME = '../data/tv-ads-kantar-meta/tv_ads_transcribed.db'
    MEDIA_FOLDER = '../data/tv-ads-kantar'

    df = pd.read_csv(DF_LOC)
    NROWS, NTHREADS = read_cmd_args(df.shape[0], 8)
    
    # df = df.iloc[:NROWS]
    df = df.iloc[4850:NROWS]
    ads = df_to_tvads(df)
    
    conn = sqlite3.connect(DB_NAME)
    
    conn.cursor().execute(
                '''CREATE TABLE IF NOT EXISTS tvads (
                    id text,
                    tone text,
                    transcript text
                )''')

    google_transcriber = FastGoogleTranscribe(
        ads, DB_NAME, MEDIA_FOLDER,NTHREADS)

    conn.close()

if __name__ == "__main__":
    main()

