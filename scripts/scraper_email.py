'''Periodically check if scraper is still live.
'''
import time
from wmp.collect import notify, webpage


def main():
    GMAIL_CRED = "../config/fred06_gmail.json"
    TO = ["fcorpuz@wesleyan.edu"]
    DB_NAME = "../data/fb-downloads/ad_multimedia.db"
    # WAIT_TIME = 120  # seconds
    WAIT_TIME = 5  # seconds
    db = webpage.FacebookDatabase(DB_NAME)
    subject = "Test FB ad scraper"
    start_time = time.time()

    _ = notify.send_message(GMAIL_CRED, TO, subject, "Started tracking")

    past_n = 0
    while True:
        current_n = db.count_ads()
        if current_n > past_n:
            past_n = current_n
            time.sleep(WAIT_TIME)
        else:
            end_time = time.time()
            run_time = end_time - start_time
            run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
            body = (
                f"No. of Ads: {current_n}\n"
                f"Stop Time: {time.ctime()}\n"
                f"Scraper ran for {run_time_str} hours"
            )
            _ = notify.send_message(GMAIL_CRED, TO, subject, body)
            break


if __name__ == "__main__":
    main()
