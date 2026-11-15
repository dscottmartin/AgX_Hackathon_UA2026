"""Send drift-risk notifications through ntfy.sh."""

import re

import requests

from rules import CAUTION, STOP


def send_ntfy_alert(topic: str, status: str, orchard_name: str, timestamp,
                    reasons: str, source: str) -> None:
    """Publish one reading to a user-selected ntfy topic."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", topic):
        raise ValueError("Topic must be 1–64 letters, numbers, underscores, or hyphens.")

    priority, tags = {
        STOP: ("urgent", "rotating_light"),
        CAUTION: ("high", "warning"),
    }.get(status, ("default", "white_check_mark"))
    message = (f"Orchard: {orchard_name}\n"
               f"Time: {timestamp:%Y-%m-%d %H:%M %Z}\n"
               f"Source: {source}\n"
               f"Status: {status}\n"
               f"Reasons: {reasons}")
    response = requests.post(
        f"https://ntfy.sh/{topic}",
        data=message.encode("utf-8"),
        headers={"Title": f"Spray or Wait: {status}", "Priority": priority, "Tags": tags},
        timeout=10,
    )
    response.raise_for_status()
