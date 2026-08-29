"""Streamlit entry point for the digital well-being app."""

from __future__ import annotations

import streamlit as st
from ai_engine import (
    analyze_screen_time,
    get_event_recommendations,
    invoke_graph_state,
    last_assistant_reply,
)
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(
    page_title="Digital Well-Being",
    page_icon="🌿",
    layout="centered",
)

THEMES = {
    "Pastel Sage 🌿": {
        "bg": "#EFEFE8",
        "sidebar_bg": "#E2E4DA",
        "primary": "#6B8E78",
        "text": "#2C3E35"
    },
    "Pastel Rose 🌸": {
        "bg": "#FDF0F0",
        "sidebar_bg": "#F7E1E1",
        "primary": "#D88A8A",
        "text": "#4A2E2E"
    },
    "Pastel Ocean 🌊": {
        "bg": "#EEF5F6",
        "sidebar_bg": "#D6E6E8",
        "primary": "#5C8D89",
        "text": "#213537"
    },
    "Pastel Lavender 💜": {
        "bg": "#F5F3F8",
        "sidebar_bg": "#E8E2F0",
        "primary": "#8E7CC3",
        "text": "#312643"
    },
    "Pastel Sunset 🍊": {
        "bg": "#FFF5EC",
        "sidebar_bg": "#FFE5D4",
        "primary": "#E07A5F",
        "text": "#3D261D"
    }
}


def apply_custom_theme(theme_name: str) -> None:
    """Seçilen temayı CSS olarak sayfaya uygular."""
    theme = THEMES.get(theme_name, THEMES["Pastel Sage 🌿"])
    css = f"""
        <style>
        .stApp {{
            background-color: {theme['bg']};
            color: {theme['text']};
        }}
        [data-testid="stSidebar"] {{
            background-color: {theme['sidebar_bg']};
        }}
        .stButton > button {{
            background-color: {theme['primary']} !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 500 !important;
        }}
        .stButton > button:hover {{
            opacity: 0.9 !important;
        }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _analysis_text(analysis: object) -> str:
    if isinstance(analysis, dict):
        insight_text = str(analysis.get("insight") or analysis)
        suggestions = analysis.get("suggestions") or []
        extra = ""
        if suggestions:
            extra = " Try this: " + "; ".join(str(item) for item in suggestions)
        elif analysis.get("suggestion"):
            extra = f" Try this: {analysis['suggestion']}"
        return f"{insight_text}{extra}".strip()
    return str(analysis or "").strip()


def _assistant_already_stored(content: str) -> bool:
    return any(
        message.get("role") == "assistant" and message.get("content") == content
        for message in st.session_state.messages
    )


def append_assistant_once(content: str) -> None:
    """Store the bot reply once in chat history."""
    text = (content or "").strip()
    if not text or _assistant_already_stored(text):
        return
    st.session_state.messages.append({"role": "assistant", "content": text})


def _graph_history() -> list:
    history: list = []
    for message in st.session_state.messages:
        content = message.get("content") or ""
        if message.get("role") == "user":
            history.append(HumanMessage(content=content))
        elif message.get("role") == "assistant":
            history.append(AIMessage(content=content))
    return history


def _resolved_user_city() -> str:
    """Prefer chat-persisted city; fall back to sidebar input."""
    return (
        st.session_state.get("user_city")
        or st.session_state.get("city_input")
        or ""
    ).strip()


def init_session_state() -> None:
    if "city" in st.session_state and not st.session_state.get("user_city"):
        st.session_state.user_city = st.session_state.pop("city")

    defaults: dict[str, object] = {
        "step": "onboarding",  # 'onboarding' veya 'dashboard'
        "messages": [],
        "last_analysis": None,
        "last_events": None,
        "has_analyzed": False,
        "user_city": "",
        "hobbies": "",
        "theme": "Pastel Sage 🌿",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    with st.sidebar:
        if st.session_state.step == "dashboard":
            if st.button("🔄 Restart / Upload New Screenshot", use_container_width=True):
                st.session_state.step = "onboarding"
                st.session_state.messages = []
                st.session_state.last_analysis = None
                st.session_state.has_analyzed = False
                st.rerun()

        st.markdown("---")

        st.caption(
            "⚠️ **Disclaimer:** This application is provided for awareness and digital well-being guidance purposes only. "
            "The generated recommendations are created by artificial intelligence and do not constitute professional "
            "medical/psychological advice. Uploaded data is not stored after processing."
        )

        st.markdown("---")

        st.selectbox(
            "Theme 🎨",
            options=list(THEMES.keys()),
            key="theme",
            help="Select a pastel theme for the interface."
        )


def render_onboarding_page() -> None:
    """İlk giriş ekranı: Ekran süresi verilerini alma."""
    st.title("Digital Well-Being 🌿")
    st.write("Welcome! Let's start by looking at your screen time habits.")

    st.markdown("---")
    st.subheader("📊 Step 1: Upload Your Screen Time Data")

    image_file = st.file_uploader(
        "Upload a screen-time screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        help="Optional. A screenshot from your phone's screen-time report.",
        key="screen_time_upload",
    )
    notes = st.text_area(
        "Anything you noticed about your usage?",
        placeholder="e.g. I was on Instagram a lot after 10pm…",
        key="usage_notes",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📊 Analyze & Continue", type="primary", use_container_width=True):
            with st.spinner("Analyzing your screen time..."):
                analysis = analyze_screen_time(image_file, notes)
                insight_text = _analysis_text(analysis)
                st.session_state.last_analysis = insight_text
                st.session_state.has_analyzed = True
                append_assistant_once(insight_text)
                st.session_state.step = "dashboard"
                st.toast("Veriler başarıyla alındı ve analiz edildi! 🎉")
                st.rerun()

    with col2:
        if st.button("Skip for now ➡️", use_container_width=True):
            st.session_state.has_analyzed = False
            st.session_state.last_analysis = None
            st.session_state.step = "dashboard"
            st.rerun()


def render_analysis_card() -> None:
    analysis = st.session_state.last_analysis
    if not analysis:
        return

    st.subheader("Latest screen-time snapshot")
    st.markdown(analysis)


def render_chat() -> None:
    st.subheader("Well-being coach")

    snapshot = (st.session_state.last_analysis or "").strip()
    for message in st.session_state.messages:
        # Snapshot card already shows the analysis; don't paint it twice in chat.
        if (
            message.get("role") == "assistant"
            and snapshot
            and (message.get("content") or "").strip() == snapshot
        ):
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about your habits, or how to unplug…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = _graph_history()[:-1]
    with st.spinner("Thinking..."):
        result = invoke_graph_state(
            HumanMessage(content=prompt),
            history=history,
            has_analyzed=bool(
                st.session_state.has_analyzed or st.session_state.last_analysis
            ),
            city=_resolved_user_city(),
        )
        city = (result.get("city") or "").strip()
        if city:
            st.session_state.user_city = city
        reply = last_assistant_reply(result)
    append_assistant_once(reply)
    with st.chat_message("assistant"):
        st.markdown(reply)


def main() -> None:
    init_session_state()
    apply_custom_theme(st.session_state.theme)

    render_sidebar()

    # Sayfa Yönlendirme Mantığı (State-Based Routing)
    if st.session_state.step == "onboarding":
        render_onboarding_page()
    else:
        st.title("Digital Well-Being Dashboard")
        if st.session_state.get("has_analyzed") or st.session_state.get("last_analysis"):
            st.success("Veriler alındı! Koçunuz sorularınızı yanıtlamaya hazır.")
        render_analysis_card()
        render_chat()


if __name__ == "__main__":
    main()