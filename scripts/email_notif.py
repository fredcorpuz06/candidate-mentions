import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
import mimetypes
import re

def send_message(user, pw, to, subject, body, doc=None):
    '''Sends email w/ optional attachment.

    Args:
    user: gmail username
    ps: password
    to: list of email addresses of recepients
    subject: text in subject
    body: text in body
    doc: filepath to attached doc

    '''
    # Buid message piece by piece
    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = ', '.join(to)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain')) # plain body text

    # One .csv Attachment
    if doc != None:
        match = re.search(r'[^\\]*\.[a-z]{3}$', str(doc))
        doc_name = doc[match.start():match.end()]
        with open(doc, 'rb') as f:
            att = MIMEBase('application', 'csv')
            att.set_payload(f.read())
            encoders.encode_base64(att)
            att.add_header('Content-Disposition', 'attachment', filename=doc_name)
            msg.attach(att)


    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls() ## Upgrade to secure connection
        server.login(user, pw)
        server.sendmail(user,to, msg.as_string()) ## to needs to be a list
        server.close()
        print('Email sent!')
    except Exception as e:
        print(e)

def notify_self_default(subject, body):
    '''Sends emails from/to my default Gmail accounts.
    '''
    with open('../config/fred06_gmail.txt', 'r') as f:
        gmail = f.read().split('\n')
    un = gmail[0].strip()
    pw = gmail[1].strip()
    to = ['fcorpuz@wesleyan.edu']

    send_message(un, pw, to, subject, body)

def main():
    subject = 'Default test email'
    body = 'Test\n Hey, what\'s up?\n\n - Fred'

    notify_self_default(subject, body)


if __name__ == "__main__":
    main()






