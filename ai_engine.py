"""LangGraph digital well-being workflow with hierarchical routing."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Fetch and strip any hidden spaces
tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()

if not tavily_key:
    raise ValueError("TAVILY_API_KEY is empty or invalid in .env file.")

from langchain_tavily import TavilySearch

search_tool = TavilySearch(
    max_results=3,
    tavily_api_key=tavily_key
)

import base64
from io import BytesIO
from typing import Annotated, Any, Literal, TypedDict

from PIL import Image
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MODEL_NAME = "gpt-4o"
REJECT_REPLY = (
    "I am designed to assist only with screen time, time management, digital "
    "balance, and finding local activities to replace screen time. How can I "
    "help you today?"
)

ANALYZE_SCREEN_TIME_SYSTEM_PROMPT = """
Role: You are an empathetic, non-judgmental Digital Well-Being Assistant.

Technique & Instructions (Step-by-Step):
Step 1: Extract total usage time and top used apps from the input screenshot or text.
Step 2: Calculate constructive real-world equivalents without inducing guilt (e.g., 'In 4.5 hours, you could watch 2 feature films, read 90 book pages, or attend a local theater play.').
Step 3: Offer a practical digital-detox or time-management suggestion. Do not ask for city or indoor/outdoor preference; a later step collects that.

Ethical Constraints & Guardrails:
- DO NOT judge, shame, or use harsh language regarding screen time.
- DO NOT provide medical advice, addiction diagnosis, or mental health therapy.
- Privacy First: Do not request or store sensitive personal health data.
"""

EVENT_RECOMMENDATIONS_SYSTEM_PROMPT = """
Role: You are a local culture and activity recommender helping people replace screen time with real-world plans.

Universal anti-hallucination (all categories):
You must ONLY recommend real events, real book titles/authors, active movies, active theater plays, and genuine physical venues that are explicitly mentioned in the Tavily search context or are widely recognized real-world classics/blockbusters.

If search results are ambiguous or lack a specific live schedule for cinema, concerts, theater, books, or hiking:
- Do NEVER invent fake film names, fictional event dates, non-existent plays, or made-up venue locations.
- Instead recommend well-known, verified local venues/spots (e.g. Atlas 1948, Kadıköy Sineması, Zorlu PSM, DasDas, Belgrat Ormanı, Atatürk Kent Ormanı, iconic book cafes) and suggest checking their current schedules.

Strict category isolation:
Answer ONLY the specific activity type requested in the user's latest message.
- Books / reading → only books and reading spots.
- Movies → only movies and cinemas.
- Concerts → only concerts.
- Theater → only plays and stages.
- Hiking / outdoor → only nature walks, parks, forests.

Keep the tone encouraging and non-judgmental. Do not ask for city or preference again if they are already known.
"""

PREFERENCE_PROMPT = (
    "To give you the best screen-free recommendations, would you prefer indoor "
    "activities (like movies, theater, or reading spots) or outdoor activities "
    "(like nature walks or concerts)? Also, which city are you located in?"
)

SCOPE_ROUTER_PROMPT = """
You are a conversation-aware scope classifier for a digital well-being assistant.
Read the FULL conversation (system context + recent history), not only the latest user line.
Reply with exactly one token: IN_SCOPE or OUT_SCOPE.

Replacing screen time with ANY offline or non-screen activity — including home activities like reading books, historical novels, indoor hobbies, cooking, or journaling — is 100% IN_SCOPE. Do NOT trigger reject_node when the user asks for offline hobby recommendations (books, music, sports, crafts) as an alternative to screen time.

IN_SCOPE if any of these are true:
- The thread is about screen time, phone/app usage, digital detox, time management, or real-world / at-home activities that replace screens.
- The user wants to stay home and read, cook, journal, craft, listen to music, or otherwise unplug.
- The assistant just asked a follow-up (city, hobbies, indoor/outdoor, location) and the user is answering it.
- The latest user message would look unrelated in isolation but continues this well-being thread
  (e.g. "I want to stay home and read historical novels" or "I live in İstanbul and I like going to cinema").

OUT_SCOPE only if the user clearly starts a new topic that is NOT an offline alternative to screens
(e.g. write my homework code, sports match trivia with no unplug intent, unrelated academic history quiz)
AND they are not answering an assistant follow-up in this conversation.
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
    "şehir",
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


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.4,
    api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
)
router_llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
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
        [SystemMessage(content=system_prompt), *_history_for_router(state)]
    )
    token = _message_text(response).strip().split()[0].upper().replace("-", "_")
    return token if token in valid else default


def _user_conversation_text(state: State) -> str:
    return "\n".join(
        _message_text(message)
        for message in state.get("messages", [])
        if _is_human(message)
    )


def _extract_city(state: State) -> str:
    blob = _user_conversation_text(state).lower()
    for needle, label in CITY_ALIASES.items():
        if needle in blob:
            return label
    return (state.get("city") or "").strip()


def _extract_activity_preference(state: State) -> str:
    blob = _user_conversation_text(state).lower()
    indoor = any(cue in blob for cue in INDOOR_CUES)
    outdoor = any(cue in blob for cue in OUTDOOR_CUES)
    if indoor and outdoor:
        return "both"
    if indoor:
        return "indoor"
    if outdoor:
        return "outdoor"
    return (state.get("activity_preference") or "").strip()


def _latest_focus_category(state: State) -> str:
    latest = _latest_user_text(state).lower()
    if any(cue in latest for cue in HOME_READING_CUES):
        return "reading"
    if any(cue in latest for cue in ("cinema", "movie", "film", "sinema", "vizyon")):
        return "cinema"
    if any(cue in latest for cue in ("concert", "konser")):
        return "concerts"
    if any(cue in latest for cue in ("tiyatro", "theater", "theatre", "play", "psm", "dasdas")):
        return "theater"
    if any(cue in latest for cue in ("hike", "hiking", "park", "walk", "doğa", "orman", "forest")):
        return "outdoor"
    if "outdoor" in latest:
        return "outdoor"
    if "indoor" in latest:
        return "indoor"

    user_blob = _user_conversation_text(state).lower()
    if any(cue in user_blob for cue in HOME_READING_CUES):
        return "reading"
    preference = _extract_activity_preference(state)
    if preference == "outdoor":
        return "outdoor"
    if preference == "indoor":
        return "indoor"
    return "indoor"


def _preference_categories(preference: str, focus: str) -> list[str]:
    if focus == "reading":
        return ["reading"]
    if focus == "cinema":
        return ["cinema"]
    if focus == "concerts":
        return ["concerts"]
    if focus == "theater":
        return ["theater"]
    if focus == "outdoor":
        return ["outdoor"]
    if preference == "indoor":
        return ["cinema", "theater", "reading"]
    if preference == "outdoor":
        return ["concerts", "outdoor"]
    return ["cinema", "theater"]


def _category_search_queries(city: str) -> dict[str, list[str]]:
    city_q = city or "Istanbul"
    reading_queries = [
        f"{city_q} best quiet book cafes",
        "best historical fiction novels recommended",
    ]
    if city_q.lower() in {"istanbul", "i̇stanbul"}:
        return {
            "cinema": [
                "current popular movies in cinemas Istanbul 2026",
                "vizyondaki en popüler filmler istanbul Paribu Cineverse",
            ],
            "concerts": [
                "Istanbul outdoor concerts Harbiye Kucukciftlik Park 2026",
                "istanbul konser takvimi bu hafta açık hava",
            ],
            "theater": [
                "istanbul tiyatro oyunları gösteriler bu ay Zorlu PSM DasDas",
                "istanbul kitap kafe güncel sergiler müzeler",
            ],
            "outdoor": [
                "istanbul güncel açık hava yürüyüş rotaları parklar Belgrad ormanı",
            ],
            "reading": reading_queries,
        }
    return {
        "cinema": [
            f"current popular movies in cinemas {city_q} 2026",
            f"vizyondaki en popüler filmler {city_q} Paribu Cineverse",
        ],
        "concerts": [f"{city_q} outdoor concerts open-air this week 2026"],
        "theater": [f"{city_q} theater plays book cafes exhibitions this month"],
        "outdoor": [f"{city_q} nature walks parks outdoor events this week"],
        "reading": reading_queries,
    }


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


def _run_category_searches(state: State) -> tuple[str, bool, list[str]]:
    city = _extract_city(state) or "Istanbul"
    preference = _extract_activity_preference(state) or "indoor"
    focus = _latest_focus_category(state)
    categories = _preference_categories(preference, focus)
    queries = _category_search_queries(city)
    chunks: list[str] = []
    any_hits = False

    for category in categories:
        for search_query in queries.get(category, [])[:2]:
            result = _run_tavily_query(search_query)
            if not result:
                continue
            any_hits = True
            chunks.append(f"[{category}] {search_query}\n{result[:1500]}")
    return "\n\n".join(chunks), any_hits, categories


def scope_check(state: State) -> Literal["reject_node", "task_router"]:
    """Step 1: route using full conversation context, including follow-up answers."""
    if (
        _assistant_asked_for_location_or_hobbies(state)
        or _looks_like_activity_followup(state)
        or _is_offline_hobby_request(state)
        or _wants_home_reading(state)
    ):
        return "task_router"

    label = _classify(
        SCOPE_ROUTER_PROMPT,
        state,
        {"IN_SCOPE", "OUT_SCOPE"},
        "IN_SCOPE",
    )
    return "task_router" if label == "IN_SCOPE" else "reject_node"


def task_router(state: State) -> Literal["screen_time_node", "activity_preference_node"]:
    """Step 2: screen-time analysis, or collect city/preference before events."""
    if _looks_like_activity_followup(state):
        return "activity_preference_node"

    label = _classify(
        TASK_ROUTER_PROMPT,
        state,
        {"SCREEN_TIME", "ACTIVITY"},
        "SCREEN_TIME",
    )
    return "activity_preference_node" if label == "ACTIVITY" else "screen_time_node"


def reject_node(state: State) -> dict[str, list[AIMessage]]:
    return {"messages": [AIMessage(content=REJECT_REPLY)]}


def _passthrough(state: State) -> dict:
    return {}


def screen_time_node(state: State) -> dict[str, list[BaseMessage]]:
    response = llm.invoke(
        [SystemMessage(content=ANALYZE_SCREEN_TIME_SYSTEM_PROMPT), *state["messages"]]
    )
    return {"messages": [response]}


def activity_preference_node(state: State) -> dict[str, Any]:
    """Stop until the user has given both city and indoor/outdoor preference."""
    city = _extract_city(state)
    preference = _extract_activity_preference(state)
    if city and preference:
        return {
            "city": city,
            "activity_preference": preference,
            "preference_ready": True,
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


def event_finder_node(state: State) -> dict[str, list[BaseMessage]]:
    city = _extract_city(state) or "your city"
    preference = _extract_activity_preference(state) or "indoor"
    focus = _latest_focus_category(state)
    live_results, has_live_data, categories = _run_category_searches(state)
    classics = "; ".join(CLASSIC_HISTORICAL_NOVELS)
    verified = (
        "Atlas 1948, Kadıköy Sineması, Paribu Cineverse, Zorlu PSM, DasDas, "
        "Belgrat Ormanı, Atatürk Kent Ormanı, and well-known book cafes in the city"
    )

    isolation = {
        "reading": (
            "The user asked for books/reading. Suggest ONLY real book titles/authors "
            "and reading spots. No movies, concerts, or hikes."
        ),
        "cinema": (
            "The user asked for movies. Suggest ONLY cinemas and films named in search "
            "results. No invented titles."
        ),
        "concerts": "The user asked for concerts. Suggest ONLY real concert venues/events.",
        "theater": "The user asked for theater. Suggest ONLY real stages/plays.",
        "outdoor": (
            "The user asked for outdoor/nature. Suggest ONLY parks, forests, and walks."
        ),
        "indoor": (
            "The user chose indoor. Stay on movies, theater, reading spots, and indoor venues."
        ),
    }.get(focus, "Stay on the requested activity type only.")

    fallback = (
        f"Search is missing live schedules. Do not invent titles or dates. "
        f"Point the user to verified spots in {city}: {verified}. "
        "Invite them to check those venues' current programs. "
        f"If they asked for books, you may name these classics only: {classics}."
    )

    prompt_messages: list[BaseMessage] = [
        SystemMessage(content=EVENT_RECOMMENDATIONS_SYSTEM_PROMPT),
        SystemMessage(
            content=(
                f"City: {city}. Indoor/outdoor preference: {preference}. "
                f"This-turn focus: {focus}. Categories searched: {', '.join(categories)}. "
                f"{isolation} "
                "ONLY recommend real events, real book titles/authors, active movies, "
                "active theater plays, and genuine venues named in the Tavily search "
                "context or widely recognized classics/blockbusters. "
                "If results are ambiguous, recommend verified venues and their current "
                "schedules instead of inventing details."
            )
        ),
    ]
    if has_live_data:
        prompt_messages.append(
            SystemMessage(
                content=(
                    "Live Tavily search results. Use only names clearly present here. "
                    "If a title, date, play, or venue is not explicit, omit it and "
                    f"fall back to verified spots ({verified}).\n\n{live_results[:6000]}"
                )
            )
        )
    else:
        prompt_messages.append(SystemMessage(content=fallback))
    prompt_messages.extend(state["messages"])
    response = llm.invoke(prompt_messages)
    return {"messages": [response]}


def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("reject_node", reject_node)
    workflow.add_node("task_router", _passthrough)
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
        },
    )
    workflow.add_edge("screen_time_node", "activity_preference_node")
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


def invoke_graph(user_message: HumanMessage, history: list | None = None) -> str:
    messages = list(history or [])
    messages.append(user_message)
    result = well_being_graph.invoke({"messages": messages})
    return _last_assistant_text(result)


def analyze_screen_time(image_file: Any | None = None, text_input: str | None = None) -> str:
    """Analyze a screen-time screenshot and/or notes through the LangGraph workflow."""
    notes = (text_input or "").strip()
    data_url = _image_to_data_url(image_file)
    if data_url:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": notes or "Please analyze this screen-time screenshot.",
            },
            {"type": "image_url", "image_url": {"url": data_url}},
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
    return invoke_graph(user_message)


def get_event_recommendations(city: str, hobbies: str) -> str:
    """Recommend local off-screen activities through the LangGraph workflow."""
    city_label = (city or "").strip() or "Istanbul"
    hobbies_label = (hobbies or "").strip() or "cinema"
    setting = "outdoor" if any(
        cue in hobbies_label.lower() for cue in ("hike", "walk", "park", "outdoor", "nature")
    ) else "indoor"
    user_message = HumanMessage(
        content=(
            f"I prefer {setting} activities. I live in {city_label} and enjoy "
            f"{hobbies_label}. Recommend current real-world alternatives to screen time."
        )
    )
    return invoke_graph(user_message)


if __name__ == "__main__":
    history: list[BaseMessage] = []
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
        result = well_being_graph.invoke({"messages": history})
        history = list(result["messages"])
        print(f"Assistant: {_last_assistant_text(result)}\n")
