"""LangGraph digital well-being workflow with hierarchical routing."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Fetch and strip any hidden spaces
tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
google_api_key = (
    (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
)

if not tavily_key:
    raise ValueError("TAVILY_API_KEY is empty or invalid in .env file.")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is empty or invalid in .env file.")
os.environ["GOOGLE_API_KEY"] = google_api_key

from langchain_tavily import TavilySearch

search_tool = TavilySearch(
    max_results=3,
    tavily_api_key=tavily_key
)

import base64
import re
from io import BytesIO
from typing import Annotated, Any, Literal, TypedDict

from PIL import Image
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MODEL_NAME = "gemini-3.6-flash"
REJECT_REPLY = (
    "I am designed to assist only with screen time, time management, digital "
    "balance, and finding local activities to replace screen time. How can I "
    "help you today?"
)
ACTIVE_CHAT_REJECT = (
    "Let's stay focused on screen time and your digital detox plan. "
    "How else can I help with your usage or offline activity ideas?"
)

ANALYZE_SCREEN_TIME_SYSTEM_PROMPT = """
Role: You are an empathetic, non-judgmental Digital Well-Being Assistant with vision.
Always reply in English.

If a screen-time screenshot (image) is provided, you MUST perform Image-to-Text extraction first. Do not skip or invent numbers that are not visible.

Structured extraction (required):
1. Total active screen hours as shown on the screenshot (or notes).
2. The top 3-4 apps with their specific durations (e.g., Instagram: 2h 30m, TikTok: 1h 45m).
3. Constructive real-world time equivalents based on those exact totals, without inducing guilt
   (e.g., 'In 4.5 hours, you could watch 2 feature films, read 90 book pages, or attend a local theater play.').
4. A practical digital-detox suggestion tailored to the specific apps you extracted.

Reply in clear markdown the UI can display:
**Total screen time:** ...
**Top apps:**
- App: duration
**Real-world equivalent:** ...
**Detox suggestion:** ...

If only text notes are provided, extract the same structure from the notes.

After the structured breakdown, ALWAYS end with this exact open-ended question (do not replace it with an indoor/outdoor binary):
To make the most of your offline time, what kind of activity would you like to do today/this weekend, and which city are you located in?

Do not recommend specific local events, movies, or venues in this step.

Ethical Constraints & Guardrails:
- DO NOT judge, shame, or use harsh language regarding screen time.
- DO NOT provide medical advice, addiction diagnosis, or mental health therapy.
- Privacy First: Do not request or store sensitive personal health data.
"""

EVENT_RECOMMENDATIONS_SYSTEM_PROMPT = """
Role: You are a hybrid local-activity guide for digital well-being: live Tavily facts plus creative, personalized reasoning.
Always reply in English.

Hybrid method:
1. Run from the Tavily search payload first. Ground artist names, event titles, dates, times, and venues in that payload.
2. Cover the user's category dynamically: cinema, concerts, music festivals, theater, stand-up, art exhibitions, outdoor workshops, or whatever they named.
3. For each recommendation, write a short structured card:
   - Artist / Event Name
   - Exact Date & Time (only if present in Tavily; otherwise say the listing did not include a time)
   - Venue / Location (e.g. Harbiye Open Air, Zorlu PSM, Parkorman)
   - Digital detox note: one sentence on why this live event is a strong screen-free swap
4. Do NOT repeat or regenerate the earlier screen-time analysis.

If search results are ambiguous or lack a specific live schedule:
- Do NEVER invent fake event names, fictional dates, or made-up venue locations.
- Recommend ONLY items that appear in the Tavily search payload. Do not use memory for current listings.
- If a date, time, or venue is missing in Tavily, omit that field rather than guessing.

Keep the tone encouraging and non-judgmental. Invite another follow-up at the end.
"""

PREFERENCE_PROMPT = (
    "To make the most of your offline time, what kind of activity would you like "
    "to do today/this weekend, and which city are you located in?"
)

SCOPE_ROUTER_PROMPT = """
You are a conversation-aware scope classifier for a digital well-being assistant.
Read the FULL conversation (system context + recent history), not only the latest user line.
Reply with exactly one token: IN_SCOPE or OUT_SCOPE.

Replacing screen time with ANY offline or non-screen activity — including home activities like reading books, historical novels, indoor hobbies, cooking, or journaling — is 100% IN_SCOPE. Do NOT trigger reject_node when the user asks for offline hobby recommendations (books, music, sports, crafts) as an alternative to screen time.

IN_SCOPE if any of these are true:
- The thread is about screen time, phone/app usage, digital detox, time management, or real-world / at-home activities that replace screens.
- The user sent a screen-time / Digital Wellbeing / app-usage UI screenshot or notes about usage hours.
- The user wants to stay home and read, cook, journal, craft, listen to music, or otherwise unplug.
- The assistant just asked a follow-up (city, hobbies, indoor/outdoor, location) and the user is answering it.

OUT_SCOPE if:
- An attached image is not a screen-time report or device-usage UI screenshot (e.g. food, selfies, random photos).
- The user clearly starts a new topic that is NOT an offline alternative to screens
  (e.g. write my homework code) AND they are not answering an assistant follow-up.
"""

IMAGE_SCOPE_PROMPT = """
Classify the attached image for a digital well-being app.
Reply with exactly one token: SCREEN_TIME_IMAGE or IRRELEVANT_IMAGE.

SCREEN_TIME_IMAGE = phone or computer Screen Time / Digital Wellbeing / app usage / battery usage dashboards and similar UI screenshots. Label these IN_SCOPE.
IRRELEVANT_IMAGE = anything else (food, people, landscapes, memes, unrelated documents). Label these OUT_SCOPE.
"""

TASK_ROUTER_PROMPT = """
You are a conversation-aware task router. Read the FULL conversation, not only the latest user line.
Reply with exactly one token: SCREEN_TIME or ACTIVITY.

ACTIVITY if any of these are true:
- The user wants events, hobbies, cinema, hiking, parks, exhibitions, or local things to do instead of screens.
- The user wants an at-home offline alternative: reading, historical novels, books, cooking, journaling, crafts, music, or sports as a screen-free plan.
- The user just shared a city, location, or hobby preference (e.g. İstanbul, cinema, walking, stay home and read).
- The previous assistant message asked for city, hobbies, or activity preferences.

SCREEN_TIME if they want usage analysis, app hours, detox, or time-management advice and are not providing location/hobby details for an activity search.

When in doubt after a location/hobby follow-up, choose ACTIVITY.
"""

FOLLOWUP_CUES = (
    "city",
    "cities",
    "hobby",
    "hobbies",
    "location",
    "where do you live",
    "share your",
    "could you share",
    "what do you enjoy",
    "to give you the best",
    "reading spots",
    "nature walks or concerts",
    "indoor",
    "outdoor",
    "which city",
    "which city are you located",
    "offline time",
    "today/this weekend",
    "şehir",
    "sehirdesiniz",
    "şehirdesiniz",
    "hangi şehirdesiniz",
    "ekran süreniz dışındaki",
    "nasıl bir aktivite",
    "bugün/bu hafta sonu",
    "hobi",
    "activity",
    "activities",
)

ACTIVITY_CUES = (
    "cinema",
    "movie",
    "film",
    "theater",
    "theatre",
    "hiking",
    "hike",
    "park",
    "walk",
    "indoor",
    "outdoor",
    "istanbul",
    "i̇stanbul",
    "live in",
    "hobby",
    "hobbies",
    "event",
    "concert",
    "konser",
    "tiyatro",
    "exhibition",
    "museum",
    "read",
    "reading",
    "novel",
    "book",
    "kitap",
    "home",
    "evde",
    "journal",
    "cooking",
    "craft",
)


CATEGORY_CUES: dict[str, tuple[str, ...]] = {
    "cinema": ("cinema", "movie", "film", "sinema", "vizyon", "box office"),
    "concerts": ("concert", "konser", "music", "müzik", "muzik", "harbiye", "festival"),
    "theater": ("tiyatro", "theatre", "play", "stage", "sahne", "psm", "dasdas", "zorlu"),
    "outdoor": (
        "hike",
        "hiking",
        "park",
        "walk",
        "outdoor",
        "sergi",
        "exhibition",
        "museum",
        "doğa",
        "doga",
        "nature",
        "yürüyüş",
        "yuruyus",
    ),
}

ALL_EVENT_CATEGORIES = ("cinema", "concerts", "theater", "outdoor")


INDOOR_CUES = (
    "indoor",
    "cinema",
    "movie",
    "film",
    "sinema",
    "theater",
    "theatre",
    "tiyatro",
    "book cafe",
    "bookcafé",
    "kitap",
    "exhibition",
    "sergi",
    "museum",
    "müze",
    "muze",
    "read",
    "reading",
    "novel",
    "book",
    "kitap",
    "stay home",
    "at home",
    "evde",
    "journal",
    "journaling",
    "cooking",
    "craft",
)
OUTDOOR_CUES = (
    "outdoor",
    "nature",
    "hike",
    "hiking",
    "park",
    "walk",
    "open-air",
    "open air",
    "açık hava",
    "acik hava",
    "doğa",
    "doga",
    "belgrad",
    "harbiye",
)
HOME_READING_CUES = (
    "read",
    "reading",
    "novel",
    "novels",
    "book",
    "books",
    "kitap",
    "historical fiction",
    "tarihi roman",
    "stay home",
    "at home",
    "evde",
    "journal",
    "journaling",
)

CLASSIC_HISTORICAL_NOVELS = (
    "Wolf Hall (Hilary Mantel)",
    "The Name of the Rose (Umberto Eco)",
    "All the Light We Cannot See (Anthony Doerr)",
    "The Pillars of the Earth (Ken Follett)",
    "Memoirs of a Geisha (Arthur Golden)",
    "The Book Thief (Markus Zusak)",
)

CITY_ALIASES = {
    "istanbul": "Istanbul",
    "i̇stanbul": "Istanbul",
    "ankara": "Ankara",
    "izmir": "Izmir",
    "i̇zmir": "Izmir",
}


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    city: str
    activity_preference: str
    preference_ready: bool
    screen_time_done: bool
    has_analyzed: bool


llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.2,
    google_api_key=google_api_key,
)
router_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.2,
    google_api_key=google_api_key,
)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _parse_data_url(value: str) -> tuple[str, str] | None:
    if not value.startswith("data:") or ";base64," not in value:
        return None
    header, encoded = value.split(";base64,", 1)
    mime = header[5:].strip() or "image/png"
    encoded = encoded.strip()
    if not encoded:
        return None
    return mime, encoded


def _gemini_image_part(part: dict[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI-style and media image blocks for Gemini Vision."""
    part_type = str(part.get("type") or "")
    mime = str(part.get("mime_type") or part.get("mime") or "").strip()
    raw_data = part.get("data") or part.get("base64")
    image_url = part.get("image_url") or part.get("url")

    if isinstance(image_url, dict):
        image_url = image_url.get("url") or image_url.get("uri") or ""
    url = str(image_url or "").strip()

    if url:
        parsed = _parse_data_url(url)
        if parsed:
            mime, raw_data = parsed
            url = f"data:{mime};base64,{raw_data}"
        return {"type": "image_url", "image_url": url}

    if raw_data:
        if isinstance(raw_data, bytes):
            raw_data = base64.b64encode(raw_data).decode("utf-8")
        mime = mime or "image/png"
        return {"type": "image_url", "image_url": f"data:{mime};base64,{raw_data}"}

    if part_type in {"image_url", "image", "media"}:
        return part
    return part


def _normalize_content_for_gemini(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    normalized: list[Any] = []
    for item in content:
        if isinstance(item, dict) and (
            item.get("type") in {"image_url", "image", "media"}
            or "image_url" in item
            or str(item.get("mime_type") or "").startswith("image/")
        ):
            normalized.append(_gemini_image_part(item))
        else:
            normalized.append(item)
    return normalized


def _clone_message_with_content(message: Any, content: Any) -> Any:
    if isinstance(message, HumanMessage):
        return HumanMessage(content=content)
    if isinstance(message, SystemMessage):
        return SystemMessage(content=content)
    if isinstance(message, AIMessage):
        return AIMessage(content=content)
    return message


def _messages_for_gemini(messages: list[Any]) -> list[Any]:
    ready: list[Any] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                updated = dict(message)
                updated["content"] = _normalize_content_for_gemini(content)
                ready.append(updated)
            else:
                ready.append(message)
            continue
        content = getattr(message, "content", None)
        if isinstance(content, list):
            ready.append(_clone_message_with_content(message, _normalize_content_for_gemini(content)))
        else:
            ready.append(message)
    return ready


def _is_human(message: Any) -> bool:
    if isinstance(message, HumanMessage) or getattr(message, "type", None) in {"human", "user"}:
        return True
    return isinstance(message, dict) and message.get("role") == "user"


def _is_ai(message: Any) -> bool:
    if isinstance(message, AIMessage) or getattr(message, "type", None) == "ai":
        return True
    return isinstance(message, dict) and message.get("role") == "assistant"


def _latest_user_text(state: State) -> str:
    for message in reversed(state.get("messages", [])):
        if _is_human(message):
            return _message_text(message)
    if state.get("messages"):
        return _message_text(state["messages"][-1])
    return ""


def _last_assistant_before_user(state: State) -> str:
    for message in reversed(state.get("messages", [])):
        if _is_human(message):
            continue
        if _is_ai(message):
            return _message_text(message)
    return ""


def _history_for_router(state: State, max_messages: int = 16) -> list[Any]:
    return list(state.get("messages", []))[-max_messages:]


def _assistant_asked_for_location_or_hobbies(state: State) -> bool:
    previous = _last_assistant_before_user(state).lower()
    if not previous or previous.strip() == REJECT_REPLY.lower():
        return False
    return any(cue in previous for cue in FOLLOWUP_CUES)


def _looks_like_activity_followup(state: State) -> bool:
    latest = _latest_user_text(state).lower()
    if _assistant_asked_for_location_or_hobbies(state):
        return True
    if _wants_home_reading(state) or _is_offline_hobby_request(state):
        return True
    return any(cue in latest for cue in ACTIVITY_CUES)


def _wants_home_reading(state: State) -> bool:
    latest = _latest_user_text(state).lower()
    return any(cue in latest for cue in HOME_READING_CUES)


def _is_offline_hobby_request(state: State) -> bool:
    blob = f"{_latest_user_text(state)}\n{_user_conversation_text(state)}".lower()
    cues = HOME_READING_CUES + (
        "offline",
        "screen-free",
        "screen free",
        "instead of",
        "cooking",
        "craft",
        "crafts",
        "journaling",
        "hobby",
        "hobbies",
        "music",
        "sports",
    )
    return any(cue in blob for cue in cues)


def _classify(system_prompt: str, state: State, valid: set[str], default: str) -> str:
    response = router_llm.invoke(
        _messages_for_gemini(
            [SystemMessage(content=system_prompt), *_history_for_router(state)]
        )
    )
    token = _message_text(response).strip().split()[0].upper().replace("-", "_")
    return token if token in valid else default


def _user_conversation_text(state: State) -> str:
    return "\n".join(
        _message_text(message)
        for message in state.get("messages", [])
        if _is_human(message)
    )


def _extract_city_from_text(text: str) -> str:
    blob = (text or "").lower()
    for needle, label in CITY_ALIASES.items():
        if needle in blob:
            return label
    return ""


ACTIVITY_ANSWER_CUES = (
    "indoor",
    "outdoor",
    "museum",
    "müze",
    "muze",
    "sergi",
    "exhibition",
    "theater",
    "theatre",
    "tiyatro",
    "coffee",
    "kahve",
    "cafe",
    "kafe",
    "workshop",
    "atölye",
    "atolye",
    "walk",
    "yürüyüş",
    "yuruyus",
    "hike",
    "hiking",
    "park",
    "doğa",
    "doga",
    "kitap",
    "read",
    "reading",
    "concert",
    "konser",
    "cinema",
    "sinema",
    "film",
    "movie",
    "craft",
    "cooking",
    "journal",
)


def _extract_preference_from_text(text: str, *, allow_freeform: bool = False) -> str:
    blob = (text or "").strip()
    if not blob:
        return ""
    remainder = blob
    for needle in sorted(CITY_ALIASES, key=len, reverse=True):
        remainder = re.sub(re.escape(needle), " ", remainder, flags=re.IGNORECASE)
    remainder = re.sub(r"\s+", " ", remainder).strip(" ,.;:-")
    lower = remainder.lower()
    if not remainder:
        return ""
    if any(cue in lower for cue in ACTIVITY_ANSWER_CUES):
        return remainder
    if allow_freeform and len(remainder) >= 4:
        return remainder
    return ""


def _latest_has_city_or_setting_cue(state: State) -> bool:
    latest = _latest_user_text(state).lower()
    if not latest:
        return False
    if _extract_city_from_text(latest):
        return True
    return any(cue in latest for cue in ACTIVITY_ANSWER_CUES)


def _analysis_in_history(state: State) -> bool:
    markers = (
        "**total screen time**",
        "total screen time:",
        "**top apps**",
        "**detox suggestion**",
        "detox suggestion:",
        "to make the most of your offline time",
        "which city are you located",
        "hangi şehirdesiniz",
        "ekran süreniz dışındaki",
        PREFERENCE_PROMPT.lower()[:48],
    )
    for message in state.get("messages", []):
        if not _is_ai(message):
            continue
        text = _message_text(message).lower()
        if any(marker in text for marker in markers):
            return True
    return False


def _has_analyzed(state: State) -> bool:
    if state.get("has_analyzed") or state.get("screen_time_done"):
        return True
    return _analysis_in_history(state)


def _extract_city(state: State) -> str:
    latest = _extract_city_from_text(_latest_user_text(state))
    if latest:
        return latest
    from_history = _extract_city_from_text(_user_conversation_text(state))
    if from_history:
        return from_history
    return (state.get("city") or "").strip()


def _extract_activity_preference(state: State) -> str:
    latest = _extract_preference_from_text(_latest_user_text(state))
    if latest:
        return latest
    from_history = _extract_preference_from_text(_user_conversation_text(state))
    if from_history:
        return from_history
    return (state.get("activity_preference") or "").strip()


def _latest_focus_category(state: State) -> str:
    latest = _latest_user_text(state).lower()
    if any(cue in latest for cue in HOME_READING_CUES):
        return "reading"
    if any(cue in latest for cue in ("museum", "müze", "muze", "sergi", "exhibition")):
        return "museum"
    if any(cue in latest for cue in ("coffee", "kahve", "cafe", "kafe")):
        return "coffee"
    if any(cue in latest for cue in ("workshop", "atölye", "atolye")):
        return "workshop"
    if any(cue in latest for cue in ("cinema", "movie", "film", "sinema", "vizyon")):
        return "cinema"
    if any(cue in latest for cue in ("stand-up", "standup", "stand up", "comedy", "komedi")):
        return "standup"
    if any(cue in latest for cue in ("festival", "festivali", "parkorman")):
        return "festivals"
    if any(cue in latest for cue in ("concert", "konser")):
        return "concerts"
    if any(cue in latest for cue in ("tiyatro", "theater", "theatre", "play", "psm", "dasdas")):
        return "theater"
    if any(cue in latest for cue in ("hike", "hiking", "park", "walk", "doğa", "orman", "forest", "yürüyüş")):
        return "outdoor"
    preference = _extract_activity_preference(state)
    return preference or "offline activities"


ISTANBUL_LIVE_CINEMA_QUERY = (
    "current movies in cinemas Istanbul August 2026 showtimes biletinial paribu cineverse"
)
CINEMA_LIVE_CUES = (
    "cinema",
    "movie",
    "movies",
    "film",
    "films",
    "sinema",
    "vizyon",
    "showtimes",
    "biletinial",
    "cineverse",
    "paribu",
)
LIVE_EVENT_CATEGORY_CUES: dict[str, tuple[str, ...]] = {
    "cinema": CINEMA_LIVE_CUES,
    "concerts": (
        "concert",
        "konser",
        "live music",
        "biletix",
        "passo",
        "harbiye",
        "artist",
    ),
    "festivals": (
        "festival",
        "festivali",
        "music festival",
        "parkorman",
        "open air festival",
    ),
    "theater": (
        "theater",
        "theatre",
        "tiyatro",
        "play",
        "psm",
        "dasdas",
        "zorlu",
    ),
    "standup": (
        "stand-up",
        "standup",
        "stand up",
        "comedy",
        "komedi",
    ),
    "exhibitions": (
        "exhibition",
        "sergi",
        "gallery",
        "art show",
        "müze",
        "muze",
        "museum",
    ),
    "workshops": (
        "workshop",
        "workshops",
        "atölye",
        "atolye",
        "outdoor workshop",
    ),
}


def _event_search_blob(state: State) -> str:
    return " ".join(
        (
            _latest_user_text(state),
            _extract_activity_preference(state),
            (state.get("activity_preference") or ""),
        )
    ).lower()


def _detect_live_event_categories(state: State) -> list[str]:
    blob = _event_search_blob(state)
    matched = [
        category
        for category, cues in LIVE_EVENT_CATEGORY_CUES.items()
        if any(cue in blob for cue in cues)
    ]
    if matched:
        return matched
    focus = _latest_focus_category(state)
    focus_map = {
        "cinema": "cinema",
        "concerts": "concerts",
        "theater": "theater",
        "museum": "exhibitions",
        "workshop": "workshops",
        "outdoor": "festivals",
    }
    if focus in focus_map:
        return [focus_map[focus]]
    if any(cue in blob for cue in ("event", "etkinlik", "tonight", "this weekend")):
        return ["concerts", "theater", "exhibitions"]
    return []


def _wants_live_cinema(state: State) -> bool:
    return "cinema" in _detect_live_event_categories(state) or _latest_focus_category(
        state
    ) == "cinema"


def _live_cinema_query(city: str) -> str:
    city_q = (city or "Istanbul").strip() or "Istanbul"
    if city_q.lower() in {"istanbul", "i̇stanbul"}:
        return ISTANBUL_LIVE_CINEMA_QUERY
    return (
        f"current movies in cinemas {city_q} August 2026 showtimes "
        "biletinial paribu cineverse"
    )


def _city_search_labels(city: str) -> tuple[str, str]:
    city_en = (city or "Istanbul").strip() or "Istanbul"
    city_tr = "İstanbul" if city_en.lower() in {"istanbul", "i̇stanbul"} else city_en
    return city_en, city_tr


def _live_queries_for_category(city: str, category: str) -> list[str]:
    city_en, city_tr = _city_search_labels(city)
    templates = {
        "cinema": [_live_cinema_query(city_en)],
        "concerts": [
            f"{city_tr} upcoming concerts 2026 famous artists dates venues Biletix Passo",
            f"{city_en} live concerts this weekend August 2026 Harbiye Parkorman",
        ],
        "festivals": [
            f"{city_tr} music festivals 2026 dates venues Parkorman Biletix Passo",
        ],
        "theater": [
            f"{city_tr} theater plays this month 2026 Zorlu PSM DasDas Biletix dates",
        ],
        "standup": [
            f"{city_en} stand-up comedy shows August 2026 dates venues Biletix",
        ],
        "exhibitions": [
            f"{city_tr} art exhibitions galleries this month 2026 Istanbul Modern Pera Museum",
        ],
        "workshops": [
            f"{city_en} outdoor workshops classes this weekend August 2026",
        ],
    }
    return list(templates.get(category, []))


def _preference_search_queries(city: str, preference: str) -> list[str]:
    city_q = city or "Istanbul"
    pref = (preference or "screen-free activities").strip()
    queries = [
        f"{city_q} {pref} this weekend live events venues 2026",
        f"{city_q} {pref} güncel yerler etkinlikler bu hafta sonu",
    ]
    lower = pref.lower()
    if any(cue in lower for cue in ("museum", "müze", "muze", "sergi", "exhibition")):
        queries.append(f"{city_q} current museum exhibitions galleries this week")
    if any(cue in lower for cue in ("coffee", "kahve", "cafe", "kafe")):
        queries.append(f"{city_q} best specialty coffee shops book cafes")
    if any(cue in lower for cue in ("workshop", "atölye", "atolye")):
        queries.append(f"{city_q} creative workshops classes this weekend")
    if any(cue in lower for cue in ("walk", "yürüyüş", "yuruyus", "doğa", "hike", "park")):
        queries.append(f"{city_q} walking trails parks nature routes")
    if any(cue in lower for cue in ("tiyatro", "theater", "theatre")):
        queries.append(f"{city_q} theater plays this week")
    if any(cue in lower for cue in ("kitap", "read", "book", "reading")):
        queries.append(f"{city_q} book cafes quiet reading spots")
    if any(cue in lower for cue in CINEMA_LIVE_CUES):
        queries.append(_live_cinema_query(city_q))
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        unique.append(query)
    return unique[:4]


def _build_live_search_queries(state: State) -> list[str]:
    city = _extract_city(state) or "Istanbul"
    preference = (
        _extract_activity_preference(state)
        or _latest_user_text(state)
        or "screen-free activities"
    )
    queries: list[str] = []
    for category in _detect_live_event_categories(state):
        queries.extend(_live_queries_for_category(city, category))
    queries.extend(_preference_search_queries(city, preference))
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        unique.append(query)
    return unique[:5]


def _clean_tavily_results(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                content = str(item.get("content") or item.get("snippet") or "").strip()
                block = "\n".join(piece for piece in (title, url, content) if piece)
                if block:
                    parts.append(block)
            else:
                text = str(item).strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
    return str(raw).strip()


def _run_tavily_query(search_query: str) -> str:
    try:
        raw = search_tool.invoke({"query": search_query})
        return _clean_tavily_results(raw)
    except Exception:
        return ""


def _run_preference_searches(state: State) -> tuple[str, bool, list[str]]:
    preference = (
        _extract_activity_preference(state)
        or _latest_user_text(state)
        or "screen-free activities"
    )
    queries = _build_live_search_queries(state)
    live_categories = _detect_live_event_categories(state)
    chunks: list[str] = []
    any_hits = False
    for index, search_query in enumerate(queries):
        result = _run_tavily_query(search_query)
        if not result:
            continue
        any_hits = True
        limit = 4000 if index == 0 and live_categories else 1800
        label = ",".join(live_categories) or preference
        chunks.append(f"[{label}] {search_query}\n{result[:limit]}")
    return "\n\n".join(chunks), any_hits, queries


def _latest_human_message(state: State) -> Any | None:
    for message in reversed(state.get("messages", [])):
        if _is_human(message):
            return message
    return None


def _message_has_image(message: Any) -> bool:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return any(
            isinstance(part, dict)
            and (
                part.get("type") in {"image_url", "image", "media"}
                or "image_url" in part
                or str(part.get("mime_type") or "").startswith("image/")
            )
            for part in content
        )
    return False


def _image_is_irrelevant(state: State) -> bool:
    message = _latest_human_message(state)
    if message is None or not _message_has_image(message):
        return False
    response = router_llm.invoke(
        _messages_for_gemini(
            [
                SystemMessage(content=IMAGE_SCOPE_PROMPT),
                message,
            ]
        )
    )
    token = _message_text(response).strip().split()[0].upper().replace("-", "_")
    return token == "IRRELEVANT_IMAGE"


def _preference_fields_ready(state: State) -> bool:
    return bool(_extract_city(state) and _extract_activity_preference(state))


def _passthrough(state: State) -> dict[str, Any]:
    return {}


def scope_check(state: State) -> Literal["reject_node", "task_router"]:
    """Reject out-of-scope turns; otherwise hand off to task_router."""
    if _image_is_irrelevant(state):
        return "reject_node"

    if (
        _latest_has_city_or_setting_cue(state)
        or _has_analyzed(state)
        or _preference_fields_ready(state)
        or _assistant_asked_for_location_or_hobbies(state)
    ):
        return "task_router"

    label = _classify(
        SCOPE_ROUTER_PROMPT,
        state,
        {"IN_SCOPE", "OUT_SCOPE"},
        "IN_SCOPE",
    )
    if label != "IN_SCOPE":
        return "reject_node"
    return "task_router"


def task_router(
    state: State,
) -> Literal["screen_time_node", "activity_preference_node", "event_finder_node"]:
    """After analysis, send text replies to events/preferences — never back to analysis."""
    if _has_analyzed(state):
        latest = _latest_user_text(state).strip()
        wants_activity = bool(
            latest
            and (
                _preference_fields_ready(state)
                or _extract_preference_from_text(latest, allow_freeform=True)
                or _extract_activity_preference(state)
                or _latest_has_city_or_setting_cue(state)
            )
        )
        if wants_activity:
            return "event_finder_node"
        return "activity_preference_node"

    missing_in_state = not (
        (state.get("city") or "").strip()
        and (state.get("activity_preference") or "").strip()
    )
    if missing_in_state and _latest_has_city_or_setting_cue(state):
        return "activity_preference_node"
    if _preference_fields_ready(state) or _assistant_asked_for_location_or_hobbies(state):
        return "activity_preference_node"
    return "screen_time_node"


def _is_active_wellbeing_chat(state: State) -> bool:
    for message in state.get("messages", []):
        if not _is_ai(message):
            continue
        text = _message_text(message).strip()
        if text and text not in {REJECT_REPLY, ACTIVE_CHAT_REJECT}:
            return True
    return False


def reject_node(state: State) -> dict[str, list[AIMessage]]:
    reply = ACTIVE_CHAT_REJECT if _is_active_wellbeing_chat(state) else REJECT_REPLY
    return {"messages": [AIMessage(content=reply)]}


def screen_time_node(state: State) -> dict[str, list[BaseMessage]]:
    """Extract screen-time stats, then ask an open-ended city/activity question."""
    prompt_messages: list[BaseMessage] = [
        SystemMessage(content=ANALYZE_SCREEN_TIME_SYSTEM_PROMPT)
    ]
    if any(_message_has_image(message) for message in state.get("messages", [])):
        prompt_messages.append(
            SystemMessage(
                content=(
                    "A screen-time screenshot is attached. Read the image carefully and "
                    "extract total active hours plus the top 3-4 apps with their exact "
                    "durations before writing equivalents or a detox suggestion. "
                    "End with this exact English question: To make the most of your "
                    "offline time, what kind of activity would you like to do "
                    "today/this weekend, and which city are you located in?"
                )
            )
        )
    prompt_messages.extend(state["messages"])
    response = llm.invoke(_messages_for_gemini(prompt_messages))
    analysis = _message_text(response).strip()
    already_asked = (
        PREFERENCE_PROMPT.lower() in analysis.lower()
        or "which city are you located" in analysis.lower()
        or "to make the most of your offline time" in analysis.lower()
        or "hangi şehirdesiniz" in analysis.lower()
        or "ekran süreniz dışındaki" in analysis.lower()
    )
    if not already_asked:
        analysis = f"{analysis}\n\n{PREFERENCE_PROMPT}"
    return {
        "messages": [AIMessage(content=analysis)],
        "has_analyzed": True,
        "screen_time_done": True,
    }


def activity_preference_node(state: State) -> dict[str, Any]:
    """Read the latest user reply for city/setting, then unlock event search."""
    latest = _latest_user_text(state)
    city = (
        _extract_city_from_text(latest)
        or (state.get("city") or "").strip()
        or _extract_city(state)
    )
    preference = (
        _extract_preference_from_text(latest, allow_freeform=True)
        or (state.get("activity_preference") or "").strip()
        or _extract_activity_preference(state)
    )
    if preference:
        city = city or "Istanbul"
        return {
            "city": city,
            "activity_preference": preference,
            "preference_ready": True,
        }

    last = state.get("messages", [])[-1] if state.get("messages") else None
    last_text = _message_text(last).lower() if last is not None else ""
    already_asked = bool(
        last is not None
        and _is_ai(last)
        and (
            "which city are you located" in last_text
            or "to make the most of your offline time" in last_text
            or "hangi şehirdesiniz" in last_text
            or "ekran süreniz dışındaki" in last_text
            or "which city" in last_text
            or "hangi şehir" in last_text
            or PREFERENCE_PROMPT.lower()[:40] in last_text
        )
    )
    if already_asked:
        return {
            "city": city,
            "activity_preference": preference,
            "preference_ready": False,
        }

    return {
        "city": city,
        "activity_preference": preference,
        "preference_ready": False,
        "messages": [AIMessage(content=PREFERENCE_PROMPT)],
    }


def route_after_preference(state: State) -> Literal["event_finder_node", "__end__"]:
    if state.get("preference_ready"):
        return "event_finder_node"
    return END


def _looks_like_analysis_message(message: Any) -> bool:
    text = _message_text(message).lower()
    return (
        "**total screen time**" in text
        or "total screen time:" in text
        or "**top apps**" in text
        or "**detox suggestion**" in text
    )


def _chat_for_event_finder(state: State) -> list[Any]:
    """Pass recent text only so event search does not replay the analysis."""
    trimmed: list[Any] = []
    for message in state.get("messages", []):
        if _is_ai(message) and _looks_like_analysis_message(message):
            continue
        if _message_has_image(message):
            note = _message_text(message).strip() or (
                "I shared a screen-time screenshot earlier."
            )
            trimmed.append(HumanMessage(content=note))
            continue
        trimmed.append(message)
    return trimmed[-8:]


def event_finder_node(state: State) -> dict[str, list[BaseMessage]]:
    city = _extract_city(state) or (state.get("city") or "").strip() or "Istanbul"
    preference = (
        _extract_activity_preference(state)
        or _extract_preference_from_text(_latest_user_text(state), allow_freeform=True)
        or _latest_user_text(state)
        or "screen-free activities"
    )
    focus = _latest_focus_category(state)
    live_results, has_live_data, queries = _run_preference_searches(state)
    live_categories = _detect_live_event_categories(state)
    structured_output = (
        "For each live event, use this exact markdown structure drawn ONLY from Tavily:\n"
        "**Artist / Event Name:** ...\n"
        "**Date & Time:** ...\n"
        "**Venue / Location:** ...\n"
        "**Digital detox note:** ...\n"
        "Skip any field that is not explicitly present in the Tavily payload. "
        "Do not invent artists, titles, dates, or venues."
    )
    classics = "; ".join(CLASSIC_HISTORICAL_NOVELS)
    verified = (
        "Atlas 1948, Kadıköy Sineması, Paribu Cineverse, Zorlu PSM, DasDas, "
        "Belgrat Ormanı, Atatürk Kent Ormanı, Istanbul Modern, Pera Museum, "
        "and well-known book cafes or specialty coffee shops in the city"
    )
    isolation = (
        f"The user asked for: {preference} (focus hint: {focus}). "
        "Recommend ONLY real venues/events that match this specific interest "
        "in their city. Do not force an indoor/outdoor binary or unrelated categories."
    )
    fallback = (
        f"Search is missing live schedules. Do not invent titles or dates. "
        f"Point the user to verified spots in {city} that fit '{preference}': {verified}. "
        "Invite them to check those venues' current programs. "
        f"If they asked for books, you may name these classics only: {classics}."
    )

    prompt_messages: list[BaseMessage] = [
        SystemMessage(content=EVENT_RECOMMENDATIONS_SYSTEM_PROMPT),
        SystemMessage(
            content=(
                f"Always reply in English. City: {city}. "
                f"User preference (use this exactly): {preference}. "
                f"Live event categories: {', '.join(live_categories) or 'general'}. "
                f"Tavily queries used: {'; '.join(queries)}. {isolation} "
                f"{structured_output} "
                "HYBRID: ground names/dates/venues in Tavily results, then add "
                "personalized digital-detox reasons. "
                "Do NOT repeat the screen-time analysis or the preference question. "
                "Return only fresh real-world recommendations for this turn. "
                "You MUST recommend only events, movies, artists, dates, and venues "
                "that appear in the Tavily search payload. Never invent or recall "
                "listings from memory. If a detail is not written in Tavily, omit it. "
                "Invite another follow-up at the end."
            )
        ),
    ]
    if has_live_data:
        cinema_rule = (
            "STRICT GROUNDING: Recommend only movies, concerts, festivals, plays, "
            "stand-up shows, exhibitions, or workshops named in this Tavily payload. "
            "Do not add titles from memory or older knowledge.\n\n"
        )
        prompt_messages.append(
            SystemMessage(
                content=(
                    "Live Tavily search results. Use only names clearly present here. "
                    f"{cinema_rule}"
                    "If a title, date, play, or venue is not explicit, omit it and "
                    f"fall back to verified spots ({verified}).\n\n{live_results[:8000]}"
                )
            )
        )
    else:
        prompt_messages.append(SystemMessage(content=fallback))
    prompt_messages.extend(_chat_for_event_finder(state))
    response = llm.invoke(_messages_for_gemini(prompt_messages))
    return {"messages": [response]}


def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("task_router", _passthrough)
    workflow.add_node("reject_node", reject_node)
    workflow.add_node("screen_time_node", screen_time_node)
    workflow.add_node("activity_preference_node", activity_preference_node)
    workflow.add_node("event_finder_node", event_finder_node)

    workflow.add_conditional_edges(
        START,
        scope_check,
        {
            "reject_node": "reject_node",
            "task_router": "task_router",
        },
    )
    workflow.add_conditional_edges(
        "task_router",
        task_router,
        {
            "screen_time_node": "screen_time_node",
            "activity_preference_node": "activity_preference_node",
            "event_finder_node": "event_finder_node",
        },
    )
    workflow.add_edge("screen_time_node", END)
    workflow.add_conditional_edges(
        "activity_preference_node",
        route_after_preference,
        {
            "event_finder_node": "event_finder_node",
            END: END,
        },
    )
    workflow.add_edge("reject_node", END)
    workflow.add_edge("event_finder_node", END)
    return workflow.compile()


well_being_graph = build_graph()


def _to_pil_image(image_file: Any) -> Image.Image | None:
    if image_file is None:
        return None
    if isinstance(image_file, Image.Image):
        return image_file
    return Image.open(image_file)


def _image_to_data_url(image_file: Any) -> str | None:
    pil_image = _to_pil_image(image_file)
    if pil_image is None:
        return None

    image_format = (pil_image.format or "PNG").upper()
    if image_format == "JPG":
        image_format = "JPEG"
    if image_format not in {"PNG", "JPEG", "WEBP"}:
        image_format = "PNG"

    buffer = BytesIO()
    pil_image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if image_format == "JPEG" else f"image/{image_format.lower()}"
    return f"data:{mime};base64,{encoded}"


def _last_assistant_text(result: State) -> str:
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) or getattr(message, "type", None) == "ai":
            return _message_text(message)
        if isinstance(message, dict) and message.get("role") == "assistant":
            return _message_text(message)
    return ""


def invoke_graph(
    user_message: HumanMessage,
    history: list | None = None,
    **state_fields: Any,
) -> str:
    messages = list(history or [])
    messages.append(user_message)
    payload: dict[str, Any] = {"messages": messages, **state_fields}
    result = well_being_graph.invoke(payload)
    return _last_assistant_text(result)


def analyze_screen_time(image_file: Any | None = None, text_input: str | None = None) -> str:
    """Extract a textual screen-time breakdown from a screenshot and/or notes.

    Returns markdown the Streamlit UI can store in session state and display.
    Calls the vision screen-time node directly so the preference gate does not
    replace the extracted breakdown.
    """
    notes = (text_input or "").strip()
    data_url = _image_to_data_url(image_file)
    if data_url:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": notes
                or (
                    "Extract total active screen hours and the top 3-4 apps with "
                    "specific durations from this screenshot."
                ),
            },
            {"type": "image_url", "image_url": data_url},
        ]
        user_message = HumanMessage(content=content)
    else:
        user_message = HumanMessage(
            content=notes
            or (
                "Help me understand my screen time and suggest a digital-detox "
                "or time-management approach."
            )
        )

    result = screen_time_node({"messages": [user_message]})
    return _message_text(result["messages"][-1])


def get_event_recommendations(city: str, hobbies: str) -> str:
    """Recommend local off-screen activities through the LangGraph workflow."""
    city_label = (city or "").strip() or "Istanbul"
    hobbies_label = (hobbies or "").strip() or "cinema"
    user_message = HumanMessage(
        content=(
            f"I live in {city_label} and want this activity: {hobbies_label}. "
            "Recommend current real-world alternatives to screen time."
        )
    )
    return invoke_graph(
        user_message,
        has_analyzed=True,
        city=city_label,
        activity_preference=hobbies_label,
        preference_ready=True,
    )


if __name__ == "__main__":
    history: list[BaseMessage] = []
    persisted: dict[str, Any] = {}
    print("Digital well-being assistant (LangGraph)")
    print("Topics: screen time, time management, digital balance, local activities.")
    print("Type q, exit, or quit to leave.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"q", "exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        result = well_being_graph.invoke(
            {
                "messages": history,
                "has_analyzed": bool(persisted.get("has_analyzed")),
                "screen_time_done": bool(persisted.get("screen_time_done")),
                "city": persisted.get("city") or "",
                "activity_preference": persisted.get("activity_preference") or "",
                "preference_ready": bool(persisted.get("preference_ready")),
            }
        )
        history = list(result["messages"])
        persisted = {
            "has_analyzed": result.get("has_analyzed"),
            "screen_time_done": result.get("screen_time_done"),
            "city": result.get("city"),
            "activity_preference": result.get("activity_preference"),
            "preference_ready": result.get("preference_ready"),
        }
        print(f"Assistant: {_last_assistant_text(result)}\n")
