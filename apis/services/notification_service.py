from datetime import datetime, timedelta, timezone
from services.notification_helper import send_notification
from services.firebase_helper import save_notification_to_firebase

def send_notification_to_user(account: dict, title: str, body: str):
    send_notification(account["deviceToken"], title, body)
    save_notification_to_firebase(body, account["code"], (datetime.now(timezone.utc) + timedelta(hours=7)).strftime('%Y-%m-%dT%H:%M:%S.000'))