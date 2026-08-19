# ============================================================================
# StormSentinel Backend — Email Sending (Resend)
# Plain REST call against Resend's API — https://api.resend.com/emails.
# No SDK dependency; `requests` is already a project dependency.
#
# VERIFIED CONSTRAINT (not an assumption): sending from the sandbox address
# onboarding@resend.dev only delivers to the email address on the Resend
# account itself. To actually notify real users, you must verify a domain
# you own in the Resend dashboard and set RESEND_FROM_EMAIL to an address
# on that domain. Until then, this will silently only reach the account
# owner's inbox — worth knowing before assuming notifications are "live".
# ============================================================================

import requests

from app.config import get_settings

RESEND_API_URL = "https://api.resend.com/emails"


def send_email(to: str, subject: str, body_text: str) -> bool:
    """Returns True if Resend accepted the send request, False otherwise."""
    settings = get_settings()
    if not settings.resend_api_key:
        return False

    from_header = f"{settings.resend_from_name} <{settings.resend_from_email}>"
    payload = {
        "from": from_header,
        "to": [to],
        "subject": subject,
        "text": body_text,
    }
    headers = {
        "Authorization": f"Bearer {settings.resend_api_key}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(RESEND_API_URL, json=payload, headers=headers, timeout=15)
        return r.status_code < 300
    except requests.RequestException:
        return False
