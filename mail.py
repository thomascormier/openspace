import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from time import time
from email.mime.image import MIMEImage
import pandas as pd
import csv

late_date=0
def update_csv_mail(email,name):
    """
    Add a new row to the csv file number_of_people.csv
    """
    with open('email.csv', 'a', newline='') as f:
        fieldnames = ['EMAIL', 'NAME']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writerow({'EMAIL': email, 'NAME': name})
    f.close()

def send_mail():
    global late_date
    now=time()

   # reading the spreadsheet
    email_list = pd.read_csv('email.csv')
    # getting the emails
    emails = email_list['EMAIL']

    sender_email = "projetiwa@gmail.com"
    receiver_email = emails[0]
    password = "pfe_team16@"

    message = MIMEMultipart("alternative")
    message["Subject"] = "Reunion demo Ippon"
    message["From"] = sender_email
    message["To"] = receiver_email

    # Create the plain-text and HTML version of your message
    text = """\
    Hello,
    The number of people admitted inside the open space has been exceeded!
    To watch the images live, please visit Live OpenSpace"""
    html = """\
    <html>
      <body>
        <p>Hello,<br></p>
        <p style="color:red">The number of people admitted inside the open space has been exceeded!<br></p>
        <p>To watch the images live, please visit <a href="http://localhost:8501">Live OpenSpace</a>.</p><br>
        <p><img src="cid:ippon" alt="logo" width="150" height="50"></p>
      </body>
    </html>
    """

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(part1)
    message.attach(part2)

    fp = open('ippon.jpg', 'rb')
    msgImage = MIMEImage(fp.read())
    fp.close()

    # Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<ippon>')
    message.attach(msgImage)


    if now >= (late_date + 300):
        late_date=now

        for i in range(len(emails)):
            # for every record get the email addresses
            email = emails[i]
            # Create secure connection with server and send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(
                    sender_email, [email], message.as_string()
                )
    else:
        print("Email already sent")