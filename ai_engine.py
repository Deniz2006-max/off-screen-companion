"""OpenAI-backed helpers for the digital well-being app."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

Instructions: Search the web for current, real-world events, movies, or activities in {city} matching {hobbies}. Provide 2-3 specific, currently available off-screen suggestions (e.g. films in theaters now, exhibitions, parks, or plays). Keep the response encouraging, warm, and concise.
"""


def _to_pil_image(image_file: Any) -> Image.Image | None:
    """Convert an uploaded file (or existing PIL image) into a PIL Image."""
    if image_file is None:
        return None
    if isinstance(image_file, Image.Image):
        return image_file
    return Image.open(image_file)


def _image_to_data_url(image_file: Any) -> str | None:
    """Encode an uploaded image as a data URL for the OpenAI vision API."""
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


def analyze_screen_time(image_file: Any | None = None, text_input: str | None = None) -> str:
    """Analyze a screen-time screenshot and/or notes with OpenAI.

    Uses role prompting, step-by-step instructions, and ethical guardrails.
    """
    user_content: list[dict[str, Any]] = []
    data_url = _image_to_data_url(image_file)
    if data_url:
        user_content.append(
            {"type": "image_url", "image_url": {"url": data_url}}
        )

    notes = (text_input or "").strip()
    if notes:
        user_text = f"User notes about their screen time:\n{notes}"
    elif data_url is None:
        user_text = (
            "No screenshot or notes were provided. Warmly invite the user to share "
            "a screen-time screenshot or a short description of their usage."
        )
    else:
        user_text = "Please analyze this screen-time screenshot."
    user_content.append({"type": "text", "text": user_text})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ANALYZE_SCREEN_TIME_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content or ""


def get_event_recommendations(city: str, hobbies: str) -> str:
    """Recommend current local off-screen activities using web search when available."""
    city_label = (city or "").strip() or "their city"
    hobbies_label = (hobbies or "").strip() or "general well-being activities"
    system_prompt = EVENT_RECOMMENDATIONS_SYSTEM_PROMPT.format(
        city=city_label,
        hobbies=hobbies_label,
    )
    user_message = (
        f"Search the web for current, real-world events, movies currently in theaters, "
        f"exhibitions, or outdoor activities in {city_label} that match these hobbies: "
        f"{hobbies_label}. Recommend 2-3 specific options that are available now."
    )

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=system_prompt,
        input=user_message,
        tools=[{"type": "web_search_preview"}],
    )
    return response.output_text or ""


if __name__ == "__main__":
    print("--- Testing Screen Time Analysis ---")
    print(analyze_screen_time(text_input="Instagram 3 hours, TikTok 1.5 hours"))
    print("\n--- Testing Web Search Event Recommendations ---")
    print(get_event_recommendations("Istanbul", "cinema and hiking"))
