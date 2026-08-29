"""Gemini-backed helpers for the digital well-being app."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-1.5-flash"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ANALYZE_SCREEN_TIME_SYSTEM_PROMPT = """
Role: You are an empathetic, non-judgmental Digital Well-Being Assistant.

Technique & Instructions (Step-by-Step):
Step 1: Extract total usage time and top used apps from the input screenshot or text.
Step 2: Calculate constructive real-world equivalents without inducing guilt (e.g., 'In 4.5 hours, you could watch 2 feature films, read 90 book pages, or attend a local theater play.').
Step 3: Ask an open-ended question inviting the user to share their city and hobbies.

Ethical Constraints & Guardrails:
- DO NOT judge, shame, or use harsh language regarding screen time.
- DO NOT provide medical advice, addiction diagnosis, or mental health therapy.
- Privacy First: Do not request or store sensitive personal health data.
"""

EVENT_RECOMMENDATIONS_SYSTEM_PROMPT = """
Role: You are a local culture and activity recommender.

Few-Shot Examples:
User: City: Istanbul, Hobbies: Cinema, Walking
Assistant: 🎬 Vizyondaki X Filmini Kadıköy Sineması'nda izleyebilirsin. 🌳 Moda Sahili'nde 45 dakikalık doğa yürüyüşü yapabilirsin.

Instructions: Provide 2-3 realistic off-screen activity suggestions for {city} matching {hobbies}. Keep the response encouraging, warm, and concise.
"""


def _to_pil_image(image_file: Any) -> Image.Image | None:
    """Convert an uploaded file (or existing PIL image) into a PIL Image."""
    if image_file is None:
        return None
    if isinstance(image_file, Image.Image):
        return image_file
    return Image.open(image_file)


def analyze_screen_time(image_file: Any | None = None, text_input: str | None = None) -> str:
    """Analyze a screen-time screenshot and/or notes with Gemini 1.5 Flash.

    Uses role prompting, step-by-step instructions, and ethical guardrails.
    """
    contents: list[Any] = []
    pil_image = _to_pil_image(image_file)
    if pil_image is not None:
        contents.append(pil_image)

    notes = (text_input or "").strip()
    if notes:
        contents.append(f"User notes about their screen time:\n{notes}")
    elif pil_image is None:
        contents.append(
            "No screenshot or notes were provided. Warmly invite the user to share "
            "a screen-time screenshot or a short description of their usage."
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=ANALYZE_SCREEN_TIME_SYSTEM_PROMPT,
        ),
    )
    return response.text or ""


def get_event_recommendations(city: str, hobbies: str) -> str:
    """Recommend local off-screen activities with few-shot prompting."""
    city_label = (city or "").strip() or "their city"
    hobbies_label = (hobbies or "").strip() or "general well-being activities"
    system_prompt = EVENT_RECOMMENDATIONS_SYSTEM_PROMPT.format(
        city=city_label,
        hobbies=hobbies_label,
    )
    user_message = f"City: {city_label}, Hobbies: {hobbies_label}"

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )
    return response.text or ""
