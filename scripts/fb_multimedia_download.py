'''Download all multimedia from Facebook Ads
'''
import argparse
import os
import pandas as pd
import random
import sys


from wmp.collect.webpage import Facebook, FacebookDatabase


def main():

    if os.name == "posix":
        WEBDRIVER_FP = "../data/chromedriver_linux64/chromedriver"
    elif os.name == "nt":
        WEBDRIVER_FP = "../data/chromedriver_win32/chromedriver.exe"
    else:
        sys.exit("Selenium Chromedrive not found")

    ADS_CSV = "../data/textsim-queries/og_ad.csv"
    FB_CREDS = "../config/fb_api.json"
    OUTPUT_FOLDER = "../data/fb-downloads"
    DB_NAME = "ad_multimedia.db"

    # Determine ad_ids to download
    ad_ids = pd.read_csv(ADS_CSV, encoding="ISO-8859-1").ad_id.tolist()
    parser = argparse.ArgumentParser(description="Download Fb ad multimedia")
    parser.add_argument("--nads", "-n", default=0, type=int)
    ad_limit = parser.parse_args().nads

    # Create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Open headless chrome + Login to Fb
    fb = Facebook(WEBDRIVER_FP, FB_CREDS, OUTPUT_FOLDER)
    db = FacebookDatabase(f"{OUTPUT_FOLDER}/{DB_NAME}")

    # Remove ads already downloaded from our queue
    done_ads = db.list_ad_ids()
    queued_ads = set(ad_ids) - set(done_ads)
    print(f"There are {len(queued_ads)} left to download!")
    if ad_limit > 0:
        queued_ads = random.sample(queued_ads, ad_limit)
        print(f"We are downloading {ad_limit} of them")

    # Go to page + extract links + take proper screenshot + download
    for ad_id in queued_ads:
        ad_multi = fb.get_multimedia(ad_id, screenshot=True)
        ad_multi_plus = fb.download_multimedia(ad_multi)

        db.insert_multimedia(ad_multi_plus)


if __name__ == "__main__":
    main()
