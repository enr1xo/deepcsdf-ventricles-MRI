import smtplib
from email.message import EmailMessage
import os
import argparse

EMAIL_ADDRESS = "davide.navarri@gmail.com"
EMAIL_PASSWORD = "zmxxfkxqsontsqay"

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default = "Training done")
parser.add_argument("--body", type=str, default = "The lion doesn't concern himself with writing the body of an email")
args = parser.parse_args()

msg = EmailMessage()
msg["From"] = EMAIL_ADDRESS
msg["To"] = EMAIL_ADDRESS
msg["Subject"] = args.subject

msg.set_content(args.body)

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.send_message(msg)
