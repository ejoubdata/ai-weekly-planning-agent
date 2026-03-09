from __future__ import annotations

import os
import base64
from pathlib import Path
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import List, Literal, Optional

import requests
from dateutil import tz
from pydantic import BaseModel, Field

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).parent

TIMEZONE = "Europe/Paris"
WEATHER_TZ = "Europe/Zurich"

LAUSANNE = {"name": "Lausanne", "lat": 46.5197, "lon": 6.6323}

# Calendar IDs
PARTNER_CALENDAR_ID = "c.ittobane@gmail.com"
FAMILY_CALENDAR_ID = "family04422174156714544449@group.calendar.google.com"

# Email recipient
TO_EMAIL = "efxjoubert@gmail.com"

# OAuth files
CREDENTIALS_CALENDAR = str(PROJECT_DIR / "credentials_calendar.json")
CREDENTIALS_GMAIL = str(PROJECT_DIR / "credentials_gmail.json")

CONFIG_DIR = Path.home() / ".config" / "week_digest"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_CALENDAR = str(CONFIG_DIR / "token_calendar.json")
TOKEN_GMAIL = str(CONFIG_DIR / "token_gmail.json")


# ============================================================
# GOOGLE AUTH HELPERS
# ============================================================

def oauth_login(flow: InstalledAppFlow):
    if hasattr(flow, "run_console"):
        return flow.run_console()
    return flow.run_local_server(port=0)


# ============================================================
# CALENDAR
# ============================================================

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

def get_calendar_service():
    creds = None

    if os.path.exists(TOKEN_CALENDAR):
        creds = Credentials.from_authorized_user_file(TOKEN_CALENDAR, CALENDAR_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_CALENDAR,
                CALENDAR_SCOPES
            )
            creds = oauth_login(flow)

        with open(TOKEN_CALENDAR, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def fetch_events_next_7_days(service, calendar_id: str, source: str):
    local_tz = tz.gettz(TIMEZONE)
    now = datetime.now(tz=local_tz)
    end = now + timedelta(days=7)

    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=now.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
        orderBy="startTime",
        maxResults=250,
    ).execute()

    out = []
    for e in events_result.get("items", []):
        start = e["start"].get("dateTime", e["start"].get("date"))
        out.append({
            "source": source,
            "summary": e.get("summary", ""),
            "location": e.get("location", ""),
            "description": (e.get("description", "") or "")[:500],
            "start": start,
        })
    return out


# ============================================================
# WEATHER
# ============================================================

def fetch_weather_next_7_days():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAUSANNE["lat"],
        "longitude": LAUSANNE["lon"],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "windspeed_10m_max",
        ],
        "timezone": WEATHER_TZ,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def summarize_weather(weather_json):
    d = weather_json["daily"]
    tmax = d["temperature_2m_max"]
    tmin = d["temperature_2m_min"]
    pprob = d["precipitation_probability_max"]
    wmax = d["windspeed_10m_max"]

    week_hi = max(tmax)
    week_lo = min(tmin)
    wet_days = sum(1 for x in pprob if x >= 50)

    return (
        f"{LAUSANNE['name']} next 7 days: "
        f"highs up to {week_hi:.0f}°C, lows down to {week_lo:.0f}°C. "
        f"Rain risk ≥50% on {wet_days} day(s). "
        f"Max wind ~{max(wmax):.0f} km/h."
    )


# ============================================================
# LLM SUMMARY
# ============================================================

Category = Literal[
    "lunch", "dinner", "birthday",
    "partner", "family",
    "work", "travel", "other"
]

class CategorizedEvent(BaseModel):
    start: str
    summary: str
    category: Category

class WeekSummary(BaseModel):
    categorized_events: List[CategorizedEvent]
    week_summary: str = Field(description="Grouped factual summary.")


def categorize_and_summarize_events(all_events: list[dict]) -> WeekSummary:
    partner_events = [e for e in all_events if e["source"] == "partner"]
    family_events = [e for e in all_events if e["source"] == "family"]
    my_events = [e for e in all_events if e["source"] == "me"]

    # Deterministic tagging
    tagged = []

    for e in partner_events:
        tagged.append(CategorizedEvent(
            start=e["start"], summary=e["summary"], category="partner"
        ))

    for e in family_events:
        tagged.append(CategorizedEvent(
            start=e["start"], summary=e["summary"], category="family"
        ))

    compact_my = [{
        "start": e["start"],
        "summary": e["summary"],
        "location": e["location"],
        "description": e["description"],
    } for e in my_events]


    # Categorize events
    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    summary_llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    structured = llm.with_structured_output(WeekSummary)

    system = (
        "Categorize into: lunch, dinner, birthday, work, travel, other. "
        "Return structured output only."
    )

    ws = structured.invoke([
        ("system", system),
        ("user", str(compact_my))
    ])

    tagged.extend(ws.categorized_events)

    # Final factual summary
    summary_text = summary_llm.invoke([
        ("system", "Provide concise factual weekly summary grouped by category."),
        ("user", str([e.dict() for e in tagged]))
    ]).content

    return WeekSummary(
        categorized_events=tagged,
        week_summary=summary_text
    )


# ============================================================
# GMAIL
# ============================================================

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def get_gmail_service():
    creds = None

    if os.path.exists(TOKEN_GMAIL):
        creds = Credentials.from_authorized_user_file(TOKEN_GMAIL, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_GMAIL,
                GMAIL_SCOPES
            )
            creds = oauth_login(flow)

        with open(TOKEN_GMAIL, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def send_email(body_text: str):
    gmail = get_gmail_service()

    msg = EmailMessage()
    msg["To"] = TO_EMAIL
    msg["Subject"] = "Weekly calendar + weather digest"
    msg.set_content(body_text)

    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    gmail.users().messages().send(userId="me", body={"raw": encoded}).execute()


# ============================================================
# MAIN
# ============================================================

def main():
    cal = get_calendar_service()

    events_me = fetch_events_next_7_days(cal, "primary", "me")
    events_partner = fetch_events_next_7_days(cal, PARTNER_CALENDAR_ID, "partner")
    events_family = fetch_events_next_7_days(cal, FAMILY_CALENDAR_ID, "family")

    all_events = events_me + events_partner + events_family

    weather = fetch_weather_next_7_days()
    weather_summary = summarize_weather(weather)

    week_summary = categorize_and_summarize_events(all_events)

    body = f"""
WEEKLY DIGEST

WEATHER
{weather_summary}

CALENDAR
{week_summary.week_summary}
"""

    send_email(body)


if __name__ == "__main__":
    main()
