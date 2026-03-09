# v2 weekly digest (LangGraph smart planner)
from __future__ import annotations

import os
import re
import base64
from pathlib import Path
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import TypedDict, List, Literal, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dateutil import parser, tz
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

# ============================================================
# 0) ENV + PATHS
# ============================================================


os.chdir("/Users/ericjoubert/Projects/ai-week-digest")

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())

load_dotenv(PROJECT_DIR / ".env")  # ensures launchd + terminal both work

TIMEZONE = "Europe/Paris"
WEATHER_TZ = "Europe/Zurich"

LAUSANNE = {"name": "Lausanne", "lat": 46.5197, "lon": 6.6323}

# Calendars
PARTNER_CALENDAR_ID = "c.ittobane@gmail.com"
FAMILY_CALENDAR_ID = "family04422174156714544449@group.calendar.google.com"

# Email recipient
TO_EMAIL = "efxjoubert@gmail.com"

# OAuth client files in project folder
CREDENTIALS_CALENDAR = str(PROJECT_DIR / "credentials_calendar.json")
CREDENTIALS_GMAIL = str(PROJECT_DIR / "credentials_gmail.json")

# Tokens stored outside repo
CONFIG_DIR = Path.home() / ".config" / "week_digest"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_CALENDAR = str(CONFIG_DIR / "token_calendar.json")
TOKEN_GMAIL = str(CONFIG_DIR / "token_gmail.json")


# ============================================================
# 1) GOOGLE AUTH HELPERS
# ============================================================

def oauth_login(flow: InstalledAppFlow):
    # run_console isn't present in all versions; local_server works on macOS
    if hasattr(flow, "run_console"):
        return flow.run_console()
    return flow.run_local_server(port=0)


# ============================================================
# 2) GOOGLE CALENDAR
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
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_CALENDAR, CALENDAR_SCOPES)
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
        endt = e["end"].get("dateTime", e["end"].get("date"))
        out.append({
            "source": source,
            "summary": e.get("summary", ""),
            "location": e.get("location", ""),
            "description": (e.get("description", "") or "")[:500],
            "start": start,
            "end": endt,
        })
    return out

    # ============================================================
# 3) WEATHER (Open-Meteo)
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
            "weathercode",
        ],
        "timezone": WEATHER_TZ,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def weathercode_to_emoji(code: int) -> str:
    if code == 0:
        return "☀️"
    elif code in (1, 2, 3):
        return "⛅" if code <= 2 else "☁️"
    elif code in (45, 48):
        return "🌫️"
    elif code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
        return "🌧️"
    elif code in (71, 73, 75, 77, 85, 86):
        return "❄️"
    elif code in (95, 96, 99):
        return "⛈️"
    return "🌡️"

def summarize_weather_1_2_lines(weather_json):
    d = weather_json["daily"]
    tmax = d.get("temperature_2m_max", [])
    tmin = d.get("temperature_2m_min", [])
    pprob = d.get("precipitation_probability_max", [])
    wmax = d.get("windspeed_10m_max", [])
    codes = d.get("weathercode", [])
    dates = d.get("time", [])

    if not tmax or not tmin:
        return f"{LAUSANNE['name']}: weather data unavailable."

    day_labels = ["M", "T", "W", "T", "F", "S", "S"]
    day_emojis = []
    for i, date_str in enumerate(dates[:7]):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_letter = day_labels[dt.weekday()]
        emoji = weathercode_to_emoji(codes[i]) if i < len(codes) else "🌡️"
        day_emojis.append(f"{day_letter} {emoji}")

    weather_week = ", ".join(day_emojis)

    wet_days = sum(1 for x in pprob if x is not None and x >= 50)
    return (
        f"{weather_week}\n"
        f"{LAUSANNE['name']} next 7 days: highs up to {max(tmax):.0f}°C, lows down to {min(tmin):.0f}°C. "
        f"Rain risk ≥50% on {wet_days} day(s); max wind ~{(max(wmax) if wmax else 0):.0f} km/h."
    )


# ============================================================
# 4) FREE SLOTS (deterministic)
# ============================================================

def parse_dt(dt_str: str, tzname=TIMEZONE) -> datetime:
    d = parser.isoparse(dt_str) if "T" in dt_str else parser.isoparse(dt_str + "T00:00:00")
    if d.tzinfo is None:
        d = d.replace(tzinfo=tz.gettz(tzname))
    return d.astimezone(tz.gettz(tzname))

def compute_free_slots(events: list[dict], days=7) -> list[dict]:
    local_tz = tz.gettz(TIMEZONE)
    now = datetime.now(tz=local_tz)
    end = now + timedelta(days=days)

    busy = []
    for e in events:
        s = parse_dt(e["start"])
        eend = parse_dt(e["end"])
        if eend > now and s < end:
            busy.append((max(s, now), min(eend, end)))

    busy.sort(key=lambda x: x[0])

    # merge overlaps
    merged = []
    for s, e in busy:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # windows: weekday 18-22; weekend 10-13 & 14-18
    slots = []
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(days):
        d = day0 + timedelta(days=i)
        wd = d.weekday()  # Mon=0..Sun=6

        windows = []
        if wd <= 4:
            windows.append((d.replace(hour=18), d.replace(hour=22)))
        else:
            windows.append((d.replace(hour=10), d.replace(hour=13)))
            windows.append((d.replace(hour=14), d.replace(hour=18)))

        for wstart, wend in windows:
            wstart = wstart.astimezone(local_tz)
            wend = wend.astimezone(local_tz)
            if wend <= now:
                continue
            wstart = max(wstart, now)

            cursor = wstart
            for bs, be in merged:
                if be <= cursor or bs >= wend:
                    continue
                if bs > cursor:
                    slots.append({"start": cursor.isoformat(), "end": bs.isoformat()})
                cursor = max(cursor, be)
            if cursor < wend:
                slots.append({"start": cursor.isoformat(), "end": wend.isoformat()})

    # keep >=45min
    out = []
    for s in slots:
        if parse_dt(s["end"]) - parse_dt(s["start"]) >= timedelta(minutes=45):
            out.append(s)
    return out[:25]


# ============================================================
# 5) LAUSANNE GUIDE TOOLS (scrape)
# ============================================================

TLG_EVENTS_CATEGORY = "https://thelausanneguide.com/category/events"

def tlg_get_latest_weekly_url() -> str:
    html = requests.get(TLG_EVENTS_CATEGORY, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    # heuristic: first /article/ link from the events category
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "/article/" in href:
            full = urljoin("https://thelausanneguide.com", href)
            if full.startswith("https://thelausanneguide.com/article/"):
                return full
    raise RuntimeError("Could not find a weekly article link on the Events category page.")

def tlg_fetch_weekly_picks(url: str) -> dict:
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else url

    lis = [li.get_text(" ", strip=True) for li in soup.select("li")]
    picks = [x for x in lis if len(x) > 25][:30]

    if not picks:
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        picks = [text[:2000]]

    return {"url": url, "title": title, "picks": picks}

@tool
def tlg_latest_weekly_url() -> str:
    """Return the URL of the latest weekly Events article from The Lausanne Guide."""
    return tlg_get_latest_weekly_url()

@tool
def tlg_weekly_picks(url: str) -> dict:
    """Fetch and extract the weekly picks list from a given Lausanne Guide weekly Events article URL."""
    return tlg_fetch_weekly_picks(url)

    # ============================================================
# 6) FACT SUMMARY (optional LLM, still “facts”)
# ============================================================

Category = Literal["me", "partner", "family"]

class EventRow(BaseModel):
    start: str
    summary: str
    source: Category

class FactsSummary(BaseModel):
    bullets: List[str] = Field(description="Factual bullets summarizing key events by day/source.")

def build_facts_summary(events: list[dict]) -> str:
    # Keep it deterministic-ish: simple bullets grouped by day; no LLM required.
    # This is intentionally minimal.
    local_tz = tz.gettz(TIMEZONE)

    # sort events
    def sort_key(e):
        return parse_dt(e["start"]).astimezone(local_tz)

    evs = sorted(events, key=sort_key)

    lines = []
    for e in evs[:60]:
        dt = parse_dt(e["start"]).astimezone(local_tz)
        day = dt.strftime("%a %d %b")
        time = dt.strftime("%H:%M") if "T" in e["start"] else "All-day"
        src = e["source"]
        title = e["summary"] or "(No title)"
        lines.append(f"- {day} {time} [{src}] {title}")
    if not lines:
        return "- (No events found in next 7 days)"
    return "\n".join(lines)


# ============================================================
# 7A) GMAIL SEND
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
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_GMAIL, GMAIL_SCOPES)
            creds = oauth_login(flow)
        with open(TOKEN_GMAIL, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

def send_email_gmail(to_email: str, subject: str, body_text: str):
    gmail = get_gmail_service()

    msg = EmailMessage()
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body_text)

    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    gmail.users().messages().send(userId="me", body={"raw": encoded}).execute()


# ============================================================
# 7B) ZAPIER MCP -> TRELLO
# ============================================================

ZAPIER_MCP_URL = os.getenv("ZAPIER_MCP_URL", "")
ZAPIER_MCP_SECRET = os.getenv("ZAPIER_MCP_SECRET", "")
ENABLE_TRELLO_PUBLISH = os.getenv("ENABLE_TRELLO_PUBLISH", "false").lower() == "true"


async def publish_recommendations_to_trello_mcp(
    recommendations_text: str,
    week_label: str,
) -> str:
    """
    Connect to Zapier MCP over HTTP, expose only Trello tools to a small agent,
    and ask it to create exactly 3 cards in the preconfigured 'Added' list.
    """
    if not ZAPIER_MCP_URL or not ZAPIER_MCP_SECRET:
        return "Trello publish skipped: missing ZAPIER_MCP_URL or ZAPIER_MCP_SECRET."

    client = MultiServerMCPClient(
        {
            "zapier": {
                "transport": "http",
                "url": ZAPIER_MCP_URL,
                "headers": {
                    "Authorization": f"Bearer {ZAPIER_MCP_SECRET}",
                },
            }
        }
    )

    tools = await client.get_tools()

    # Keep only Trello-related tools
    trello_tools = [t for t in tools if "trello" in t.name.lower()]
    if not trello_tools:
        return "Trello publish skipped: no Trello tools found on Zapier MCP server."

    trello_agent = create_react_agent(model=llm, tools=trello_tools)

    publish_prompt = f"""
You are publishing weekly activity suggestions to Trello.

Goal:
Create exactly 3 Trello cards using the Trello MCP tool already configured in Zapier.
The board and destination list are already fixed in Zapier to the correct board and the 'Added' list.

Rules:
- Create exactly 3 cards.
- Use one card per recommendation.
- Do not create extra cards.
- Keep each card title short and actionable.
- Put the reasoning, chosen slot, and source URL in the description.
- Prefix each title with the week label: [{week_label}]
- Do not invent new activities beyond the recommendations below.

Recommendations:
{recommendations_text}

After creating the cards, return a short 3-bullet confirmation with the created card titles.
"""

    res = await trello_agent.ainvoke({"messages": [("user", publish_prompt)]})
    return res["messages"][-1].content


def publish_recommendations_to_trello_sync(
    recommendations_text: str,
    week_label: str,
) -> str:
    if not ENABLE_TRELLO_PUBLISH:
        return "Trello publish disabled."

    return asyncio.run(
        publish_recommendations_to_trello_mcp(
            recommendations_text=recommendations_text,
            week_label=week_label,
        )
    )


# ============================================================
# 8) LANGGRAPH: STATE + NODES
# ============================================================

class State(TypedDict, total=False):
    events: list[dict]
    free_slots: list[dict]
    weather_summary: str
    facts_summary: str
    recommendations: str
    trello_result: str
    email_body: str

# Agent (tool-calling)
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
agent = create_react_agent(model=llm, tools=[tlg_latest_weekly_url, tlg_weekly_picks])

def node_free_slots(state: State) -> State:
    return {"free_slots": compute_free_slots(state["events"])}

def node_weather(state: State) -> State:
    w = fetch_weather_next_7_days()
    return {"weather_summary": summarize_weather_1_2_lines(w)}

def node_facts(state: State) -> State:
    return {"facts_summary": build_facts_summary(state["events"])}

def node_recommendations(state: State) -> State:
    prompt = f"""
You are a smart weekly planning assistant.

You MUST use The Lausanne Guide weekly list (tools) as the source of activities.
Goal: propose 3 ranked activity suggestions.

Priorities:
1) family-friendly
2) activities for young kids
3) original shows

Inputs:
- Weather summary: {state["weather_summary"]}
- Free slots (ISO): {state["free_slots"]}

Rules:
- Pick a suggested time slot for each recommendation from the provided free slots.
- If weather suggests rain, bias toward indoor options when possible.
- Keep output concise.
- Include the Lausanne Guide weekly article URL in the output.
- Do NOT invent events not present in the Lausanne Guide weekly list.

Output format (exact):
1) <Title> — <chosen slot start> to <slot end>
   Why: <1-2 lines>
   Source: <Lausanne Guide item paraphrase> (URL: <url>)
2) ...
3) ...
"""

    res = agent.invoke({"messages": [("user", prompt)]})
    recommendations = res["messages"][-1].content
    return {"recommendations": recommendations}

def node_trello_publish(state: State) -> State:
    local_tz = tz.gettz(TIMEZONE)
    start_date = datetime.now(tz=local_tz)
    end_date = start_date + timedelta(days=6)
    week_label = f"{start_date.strftime('%b %-d')}–{end_date.strftime('%b %-d, %Y')}"

    trello_result = publish_recommendations_to_trello_sync(
        recommendations_text=state["recommendations"],
        week_label=week_label,
    )
    return {"trello_result": trello_result}


def node_email_body(state: State) -> State:
    local_tz = tz.gettz(TIMEZONE)
    now = datetime.now(tz=local_tz).strftime("%A %d %b %Y, %H:%M %Z")

    trello_section = state.get("trello_result", "Trello publish not attempted.")

    body = f"""WEEKLY DIGEST + PLAN — generated {now}

WEATHER (next 7 days)
{state["weather_summary"]}

CALENDAR (next 7 days)
{state["facts_summary"]}

RECOMMENDATIONS (from The Lausanne Guide)
{state["recommendations"]}

TRELLO
{trello_section}
"""
    return {"email_body": body}


def build_graph():
    g = StateGraph(State)
    g.add_node("free_slots", node_free_slots)
    g.add_node("weather", node_weather)
    g.add_node("facts", node_facts)
    g.add_node("recommendations", node_recommendations)
    g.add_node("trello_publish", node_trello_publish)
    g.add_node("email", node_email_body)

    g.set_entry_point("free_slots")
    g.add_edge("free_slots", "weather")
    g.add_edge("weather", "facts")
    g.add_edge("facts", "recommendations")
    g.add_edge("recommendations", "trello_publish")
    g.add_edge("trello_publish", "email")
    g.add_edge("email", END)

    return g.compile()
    # ============================================================
# 9) MAIN
# ============================================================

def main():
    # Calendar fetch (deterministic)
    cal = get_calendar_service()
    events_me = fetch_events_next_7_days(cal, "primary", "me")
    events_partner = fetch_events_next_7_days(cal, PARTNER_CALENDAR_ID, "partner")
    events_family = fetch_events_next_7_days(cal, FAMILY_CALENDAR_ID, "family")
    all_events = events_me + events_partner + events_family

    app = build_graph()
    result = app.invoke({"events": all_events})

    # Calculate dynamic date range for email subject
    local_tz = tz.gettz(TIMEZONE)
    start_date = datetime.now(tz=local_tz)
    end_date = start_date + timedelta(days=6)
    subject = f"Weekly Digest: {start_date.strftime('%A %B %-d')} to {end_date.strftime('%A %B %-d, %Y')}"

    send_email_gmail(
        to_email=TO_EMAIL,
        subject=subject,
        body_text=result["email_body"],
    )

if __name__ == "__main__":
    main()