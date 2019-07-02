'''Send email notifications.
'''
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_message(gmail_cred, to, subject, body):
    '''Sends email w/ optional attachment.

    Parameters
    ----------
    gmail_cred:
        File loc of JSON file containing email & password keys
    to:
        List of email addresses of recepients
    subject
        Text in subject
    body
        Text in body

    Returns
    -------
    Bool flagging if email was successfully sent
    '''

    with open(gmail_cred, 'r') as f:
        gmail = json.load(f)
    un = gmail['email']
    pw = gmail['password']

    # Buid message piece by piece
    msg = MIMEMultipart()
    msg['From'] = un
    msg['To'] = ', '.join(to)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))  # plain body text

    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()  # Upgrade to secure connection
        server.login(un, pw)
        server.sendmail(un, to, msg.as_string())  # to needs to be a list
        server.close()
        print('Email sent!')
        return True
    except Exception as e:
        print(e)
        return False


def _notify_self_default(subject, body):
    '''Sends emails from/to my default Gmail accounts.
    '''
    gmail_cred = "../../../config/fred06_gmail.json"
    to = ['fcorpuz@wesleyan.edu']

    _ = send_message(gmail_cred, to, subject, body)


def main():
    subject = 'from your scraper'
    body = 'yellooo'

    _notify_self_default(subject, body)


if __name__ == "__main__":
    main()
