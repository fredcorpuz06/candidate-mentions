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

from collections import namedtuple, OrderedDict
import json
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, NoSuchAttributeException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
import sqlite3
from threading import Thread, Lock
import time
import os

from .download import FileDownload


class WebPage:

    def __init__(self, webdriver_loc, out_folder):
        self.browser = self._prep_browser(webdriver_loc)
        self.out_folder = self._create_folder(out_folder)
        self.getter = FileDownload()

    def _prep_browser(self, webdriver_loc):
        '''Opens headless chrome.

        TO DO: create parent class with common Selenium functionality
        '''
        opts = webdriver.ChromeOptions()
        opts.add_argument('headless')
        opts.add_argument("--start-maximized")
        opts.add_argument("--start-fullscreen")
        opts.add_argument(
            '--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"')
        opts.add_argument('log-level=3')  # supress STOP from Fb
        browser = webdriver.Chrome(options=opts, executable_path=webdriver_loc)
        return browser

    def _create_folder(self, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        return out_folder

    def _get_by_css(self, css):
        '''Wait for element to load'''
        wait = WebDriverWait(self.browser, self._WAIT_SECS)
        wait.until(expected_conditions.visibility_of_element_located(
            (By.CSS_SELECTOR, css)))
        return self.browser.find_element_by_css_selector(css)

    def _get_by_tag(self, tag):
        wait = WebDriverWait(self.browser, self._WAIT_SECS)
        wait.until(expected_conditions.visibility_of_element_located(
            (By.TAG_NAME, tag)))
        return self.browser.find_element_by_tag_name(tag)

    def get_link(self, css=None, tag=None):
        '''Extract link from one element specified by CSS selector or tag.'''
        assert not (css == tag == None), "No selector specified"
        try:
            if css:
                m = self._get_by_css(css)
            else:
                m = self._get_by_tag(tag)
        except (NoSuchElementException, TimeoutException):
            return None

        try:
            link = m.get_attribute("src")
        except NoSuchAttributeException:
            link = None
        finally:
            return link


class Facebook(WebPage):

    _FACEBOOK_FRONT = "https://www.facebook.com"
    _SPONSOR_CSS = "._7pg4"
    _BODY_IMG_CSS = "._7jys"
    _BODY_VID_TAG = "video"
    _WAIT_SECS = 3
    AD_FIELDS = [
        "ad_id", "sponsor", "image", "video",
        "sshot_loc", "sponsor_loc", "image_loc", "video_loc"]

    AdMulti = namedtuple("AdMulti", AD_FIELDS)

    def __init__(self, webdriver_loc, fb_creds_loc, out_folder):
        super().__init__(webdriver_loc, out_folder)
        self.fb_creds = self._get_fb_creds(fb_creds_loc)
        self._login_fb()

    def _get_fb_creds(self, fb_creds_loc):
        '''Gets provided FB credentials from JSON file.'''
        with open(fb_creds_loc, "r") as f:
            fb_creds = json.load(f)
        return fb_creds

    def _login_fb(self):
        '''Logs into Facebook using given credentials.'''
        self.browser.get(self._FACEBOOK_FRONT)
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

        # assert [l for l in [img_link, vid_link] if l != None] != [
        # ], f"--- {ad_id} Image and video links from ad body are both missing."

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

    def list_ad_ids(self):
        self.cursor.execute(
            'SELECT DISTINCT ad_id FROM multimedia;'
        )
        ad_ids = [int(record[0]) for record in self.cursor.fetchall()]
        return ad_ids

    def count_ads(self):
        self.cursor.execute(
            'SELECT COUNT(ad_id) FROM multimedia;'
        )
        return self.cursor.fetchall()[0][0]


class GoogleImages(WebPage):
    _GOOGLE_FRONT = "https://www.google.com/search?q=barack+obama&tbm=isch"
    _IMAGE_CSS = ".rg_i"
    _CONTAINER_IMG_CSS = ".irc_mi"

    def __init__(self, webdriver_loc, out_folder):
        super().__init__(webdriver_loc, out_folder)

    def _format_query(self, query_str):
        search_term = "+".join(query_str.split())
        return f"https://www.google.com/search?q={search_term}&tbm=isch"

    def _format_name(self, query_str, num=None):
        basename = "_".join(query_str.split())
        if num is not None:
            return f"{basename}_{num:03}"
        return basename

    def image_search(self, query, nimages=10):
        uri = self._format_query(query)
        base_name = self._format_name(query)
        person_folder = f"{self.out_folder}/{base_name}"
        self._create_folder(person_folder)

        self.browser.get(uri)
        self._scroll_bottom()

        img_ids = self.get_image_ids()
        img_links = OrderedDict()
        for img_id in img_ids[:nimages]:
            potential_links = self.get_image_link(uri, img_id)
            for pl in potential_links:
                img_links[pl] = ""

        for idx, (link, _) in enumerate(img_links.items()):
            self.getter.download_image(
                link, person_folder,
                self._format_name(base_name, num=idx),
                default_format="jpg")

    def _scroll_bottom(self):
        '''Scrolls to the bottom of the webpage.
        '''
        last_height = self.browser.execute_script(
            "return document.body.scrollHeight")
        while True:
            self.browser.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            new_height = self.browser.execute_script(
                "return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def get_image_ids(self):
        ms = self.browser.find_elements_by_css_selector(self._IMAGE_CSS)
        all_ids = [m.get_attribute("id") for m in ms]
        all_ids = [i for i in all_ids if (i is not "") & (i is not None)]
        return all_ids

    def get_image_link(self, uri, img_id):
        container_link = f"{uri}#imgrc={img_id}"
        self.browser.get(container_link)
        time.sleep(0.5)
        ms = self.browser.find_elements_by_css_selector(
            self._CONTAINER_IMG_CSS)
        return [m.get_attribute("src") for m in ms]
