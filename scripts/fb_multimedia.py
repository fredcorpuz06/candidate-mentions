'''
'''

from wmp.collect.webpage import Facebook, FacebookDatabase


def main():
    ADS = [
        2079670018754991,
        257927201736805,
        510736936056223,
        351529942259951,
        324402845024092,
        1521102681323111,
        2031660146918050,
        316507982480388,
        254565331977792,
        571879176568068]

    FB_CREDS = "../config/fb_api.json"
    WEBDRIVER_FP = "../data/chromedriver_win32/chromedriver.exe"
    OUTPUT_FOLDER = "../data/fb-downloads"
    DB_NAME = "ad_multimedia.db"

    # Open headless chrome + Login to Fb
    fb = Facebook(WEBDRIVER_FP, FB_CREDS, OUTPUT_FOLDER)
    db = FacebookDatabase(f"{OUTPUT_FOLDER}/{DB_NAME}")

    # Go to page + extract links + take proper screenshot + download
    for ad_id in ADS:
        ad_multi = fb.get_multimedia(ad_id, screenshot=True)
        ad_multi_plus = fb.download_multimedia(ad_multi)

        db.insert_multimedia(ad_multi_plus)


if __name__ == "__main__":
    main()
