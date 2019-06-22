import numpy as np
import pandas as pd

import extract_media
import recognize
import perceive


df = pd.read_csv('../data/og_ad_sample.csv')
print(df.head())

# Download text + screenshot + image + video 
# Input: 
# - list of ad_ids: [2079670018754991, 351529942259951, 1521102681323111]
# - output_fp: output folder path (i.e. "../output/fbDownloads")
# - fb_credentials_loc: 
# - webdriver_loc: 
# Actions: Download all content to designated folders + create SQLite DB with columns
# - ad_id
# - creative_body: text contained in ad
# - screenshot_loc: screenshots/2079670018754991_body0.jpg
# - image_loc: images/2079670018754991_img0.jpg
# - video_loc: videos/2079670018754991_vid0.mp4
# Returns:
# 

fb_page = extract_media.FbNavigator(webdriver, fb_creds)
fb_page.login()
for ad_id in ad_ids:
    fb_page.get_page_media(ad_id)
    fb_page.save_results(ad_id)

# Get GCP video transcript

# Find all text references in creative body + video transcript

# Find all image/video references

# Determine text sentiemnt

# Determine image/video sentiment