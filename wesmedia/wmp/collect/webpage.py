'''Extract multi-media data from Facebook ads.

This module contains methods for extracting text, images, videos and video
transcriptions from Facebook ads. Selenium is used to navigate the webpage of
a Facebook ad, take a screenshot, and gather the text data and all the links
to images/videos. Google Cloud's Speech-to-Text is used to transcribe the
videos.


The module structure is the following:


- The ``FbMediaFinder`` class implements ...

- The ``MediaDownloader`` class

- 

Code inspired by: 
https://www.techbeamers.com/selenium-webdriver-python-tutorial/
https://github.com/CoreyMSchafer/code_snippets/tree/master/Python-SQLite
'''

# Authors: Frederick Corpuz

from collections import namedtuple
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, NoSuchAttributeException
import sqlite3
from threading import Thread, Lock
import json
from .download import FileDownload
FACEBOOK_FRONT = "https://www.facebook.com"


class Facebook:

    _SPONSOR_CSS = "._7pg4"
    _BODY_IMG_CSS = "._7jys"
    _BODY_VID_TAG = "video"
    AD_FIELDS = [
        "ad_id", "sponsor", "image", "video",
        "sshot_loc", "sponsor_loc", "image_loc", "video_loc"]

    AdMulti = namedtuple("AdMulti", AD_FIELDS)

    def __init__(self, webdriver_loc, fb_creds_loc, out_folder):
        self.browser = self._prep_browser(webdriver_loc)
        self.fb_creds = self._get_fb_creds(fb_creds_loc)
        self.out_folder = out_folder
        self.getter = FileDownload()
        self._login_fb()

    def _prep_browser(self, webdriver_loc):
        '''Opens headless chrome.'''
        opts = webdriver.ChromeOptions()
        opts.add_argument('headless')
        opts.add_argument("--start-maximized")
        opts.add_argument("--start-fullscreen")
        opts.add_argument('log-level=3')  # supress STOP from Fb
        browser = webdriver.Chrome(options=opts, executable_path=webdriver_loc)

        return browser

    def _get_fb_creds(self, fb_creds_loc):
        '''Gets provided FB credentials from JSON file.'''
        with open(fb_creds_loc, "r") as f:
            fb_creds = json.load(f)
        return fb_creds

    def _login_fb(self):
        '''Logs into Facebook using given credentials.'''
        self.browser.get(FACEBOOK_FRONT)
        self.browser.find_element_by_id(
            'email').send_keys(self.fb_creds['username'])
        self.browser.find_element_by_id(
            'pass').send_keys(self.fb_creds['password'])
        self.browser.find_element_by_id(
            'loginbutton').click()
        print("\n--- Logged into Facebook\n\n")

    def get_multimedia(self, ad_id, screenshot=False):
        '''Extracts links for ads images + videos.'''
        ad_link = (f"https://www.facebook.com/ads/archive/render_ad/"
                   f"?id={ad_id}&access_token={self.fb_creds['access_token']}")
        self.browser.get(ad_link)
        sponsor_link = self.get_link(self._SPONSOR_CSS)
        img_link = self.get_link(self._BODY_IMG_CSS)
        vid_link = self.get_link(self._BODY_VID_TAG)
        if screenshot:
            sshot_loc = self.body_screenshot(ad_id)

        assert [l for l in [img_link, vid_link] if l != None] != [
        ], "Image and video links from ad body are both missing."

        return self.AdMulti(
            ad_id, sponsor_link, img_link, vid_link,
            sshot_loc, None, None, None)

    def download_multimedia(self, multi):
        '''Downloads images + videos from collected links.'''

        sponsor_loc = self.getter.download_image(
            multi.sponsor, self.out_folder,
            f"{multi.ad_id}_sponsor")
        image_loc = self.getter.download_image(
            multi.image, self.out_folder,
            f"{multi.ad_id}_image")
        video_loc = self.getter.download_video(
            multi.video, self.out_folder,
            f"{multi.ad_id}_video")

        multi = multi._replace(sponsor_loc=sponsor_loc)
        multi = multi._replace(image_loc=image_loc)
        multi = multi._replace(video_loc=video_loc)

        return multi

    def get_link(self, css=None, tag=None):
        '''Extract link to single element specified by CSS selector or tag.'''
        assert not (css == tag == None), "No selector specified"
        try:
            if css:
                m = self._get_by_css(css)
            else:
                m = self._get_by_tag(tag)
        except NoSuchElementException:
            return None

        try:
            link = m.get_attribute("src")
        except NoSuchAttributeException:
            link = None
        finally:
            return link

    def _get_by_css(self, css):
        return self.browser.find_element_by_css_selector(css)

    def _get_by_tag(self, tag):
        return self.browser.find_element_by_tag_name(tag)

    def body_screenshot(self, ad_id):
        '''Takes a screenshot of the page body.'''
        body = self.browser.find_element_by_tag_name('body')
        body_data = body.screenshot_as_png

        sshot_loc = f"{self.out_folder}/{ad_id}_sshot.png"
        with open(sshot_loc, 'wb+') as f:
            f.write(body_data)

        return f"{ad_id}_sshot.png"


class FacebookDatabase:

    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.lock = Lock()
        self.fields = Facebook.AD_FIELDS
        self._create_table()

    def _create_table(self):
        '''Creates `multimedia` table in database.'''
        cols = [f"{f} text"for f in self.fields]
        all_cols = ", ".join(cols)
        with self.conn:
            self.cursor.execute(
                f'CREATE TABLE IF NOT EXISTS multimedia ({all_cols});',

            )

    def insert_multimedia(self, multi):
        '''Inserts record into `multimedia` table in database.'''
        self.lock.acquire()
        with self.conn:
            self.cursor.execute(
                'INSERT INTO multimedia VALUES(?,?,?,?,?,?,?,?);',
                multi
            )
        self.lock.release()

    def full_query(self, fields):
        fields_str = ", ".join(fields)
        self.cursor.execute(
            '''SELECT {} FROM multimedia;'''.format(fields_str)
        )
        return self.cursor.fetchall()
