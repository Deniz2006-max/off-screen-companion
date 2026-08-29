"""Placeholder AI helpers for the digital well-being app.

These functions return mock data so the Streamlit UI can be wired up
before real Gemini (google-genai) calls are added.
"""

from __future__ import annotations

from typing import Any


def analyze_screen_time(image_file: Any | None, text_input: str) -> dict[str, Any]:
    """Analyze a screen-time screenshot and/or free-text notes.

    Args:
        image_file: Uploaded image (e.g. Streamlit UploadedFile) or None.
        text_input: Optional notes from the user about their usage.

    Returns:
        A dict with mock insights. Replace the body with a Gemini call later.
    """
    _ = image_file  # unused until real vision analysis is wired up
    notes = (text_input or "").strip()

    return {
        "total_hours": 6.4,
        "top_apps": [
            {"name": "Social media", "hours": 2.8},
            {"name": "Video streaming", "hours": 1.9},
            {"name": "Messaging", "hours": 1.1},
        ],
        "insight": (
            "Most of your time is in passive scrolling and video. "
            "A short outdoor break would help reset focus."
            if not notes
            else f"Based on your note (“{notes[:80]}”), try a 20-minute offline break."
        ),
        "suggestion": "Take a 15-minute walk without your phone.",
        "source": "mock",
    }


def get_event_recommendations(city: str, hobbies: list[str] | str) -> list[dict[str, str]]:
    """Suggest local, offline events that match the user's city and hobbies.

    Args:
        city: City name used for local recommendations.
        hobbies: Hobby list or a comma-separated string.

    Returns:
        A list of mock event dicts. Replace with a Gemini / maps lookup later.
    """
    city_label = (city or "your city").strip() or "your city"
    if isinstance(hobbies, str):
        hobby_list = [h.strip() for h in hobbies.split(",") if h.strip()]
    else:
        hobby_list = [h.strip() for h in (hobbies or []) if str(h).strip()]

    primary = hobby_list[0] if hobby_list else "well-being"
    secondary = hobby_list[1] if len(hobby_list) > 1 else "community"

    return [
        {
            "title": f"{primary.title()} meetup",
            "city": city_label,
            "when": "Saturday 10:00",
            "why": f"A low-screen way to practice {primary} with others nearby.",
        },
        {
            "title": f"{secondary.title()} workshop",
            "city": city_label,
            "when": "Sunday 14:00",
            "why": f"Hands-on {secondary} session — leave the phone in your bag.",
        },
        {
            "title": "Park walk & talk",
            "city": city_label,
            "when": "Weekday evening",
            "why": "Short outdoor reset after a high screen-time day.",
        },
    ]
