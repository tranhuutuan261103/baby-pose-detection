
"""Server Side FCM sample.

Firebase Cloud Messaging (FCM) can be used to send messages to clients on iOS,
Android and Web.

This sample uses FCM to send two types of messages to clients that are subscribed
to the news topic. One type of message is a simple notification message (display message).
The other is a notification message (display notification) with platform specific
customizations. For example, a badge is added to messages that are sent to iOS devices.
"""

import argparse
import json
import requests
import google.auth.transport.requests
import os
import logging
# Configure the logger
logging.basicConfig(filename='apis/server_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

from google.oauth2 import service_account

current_directory = os.path.dirname(os.path.abspath(__file__))
service_account_path = os.path.join(current_directory, 'serviceAccountKey.json')

PROJECT_ID = 'pbl6-519c3'
BASE_URL = 'https://fcm.googleapis.com'
FCM_ENDPOINT = 'v1/projects/' + PROJECT_ID + '/messages:send'
FCM_URL = BASE_URL + '/' + FCM_ENDPOINT
SCOPES = ['https://www.googleapis.com/auth/firebase.messaging']
# DEVICE_TOKEN = 'e1ZKfBGyQba2a5bOQqk8iw:APA91bGEm1xG92Pbe6psZcdOcLJvJMWEdRgrbcuSUViV77kN5MRe5jdyE5Korp-exPilg9xtQ1LvPeh3CfuxOamV6uTzSOpYa116rAZeYBH6TETPXZ0J4JA'


# [START retrieve_access_token]
def _get_access_token():
    """Retrieve a valid access token that can be used to authorize requests.

    :return: Access token.
    """
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path, scopes=SCOPES)
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token
# [END retrieve_access_token]

def _send_fcm_message(fcm_message):
    """Send HTTP request to FCM with given message.

    Args:
        fcm_message: JSON object that will make up the body of the request.
    """
    # [START use_access_token]
    headers = {
        'Authorization': 'Bearer ' + _get_access_token(),
        'Content-Type': 'application/json; UTF-8',
    }
    # [END use_access_token]
    resp = requests.post(FCM_URL, data=json.dumps(fcm_message), headers=headers)

    if resp.status_code == 200:
        logging.info(f'Message sent to Firebase for delivery, response: {resp.text}')
        print('Message sent to Firebase for delivery, response:')
        print(resp.text)
    else:
        logging.error(f'Unable to send message to Firebase {resp.text}')
        print('Unable to send message to Firebase')
        print(resp.text)



def _build_device_message(token, title: str, body: str):
    """Construct a notification message for a specific device."""
    return {
        'message': {
            'token': token,
            'notification': {
                'title': title,
                'body': body
            }
        }
    }

def send_notification_to_device(token, title: str, body: str):
    # Sử dụng DEVICE_TOKEN đã định nghĩa sẵn để gửi thông báo
    device_message = _build_device_message(token, title, body)
    print('FCM request body for sending message to specific device:')
    print(json.dumps(device_message, indent=2))
    _send_fcm_message(device_message)