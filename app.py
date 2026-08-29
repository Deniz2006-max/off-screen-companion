"""Streamlit entry point for the digital well-being app."""

from __future__ import annotations

import streamlit as st
from ai_engine import analyze_screen_time, get_event_recommendations

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


def init_session_state() -> None:
    defaults: dict[str, object] = {
        "step": "onboarding",  # 'onboarding' veya 'dashboard'
        "messages": [],
        "last_analysis": None,
        "last_events": None,
        "city": "",
        "hobbies": "",
        "theme": "Pastel Sage 🌿",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Your context")
        st.text_input("City", placeholder="e.g. Istanbul", key="city")
        st.text_input(
            "Hobbies",
            placeholder="e.g. hiking, photography, chess",
            key="hobbies",
        )

        st.markdown("---")

        if st.button("🎨 Suggest Local Events", use_container_width=True):
            events = get_event_recommendations(
                st.session_state.city,
                st.session_state.hobbies,
            )
            st.session_state.last_events = events
            st.session_state.messages.append(
                {"role": "assistant", "content": events}
            )

        if st.session_state.step == "dashboard":
            if st.button("🔄 Restart / Upload New Screenshot", use_container_width=True):
                st.session_state.step = "onboarding"
                st.rerun()

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
                st.session_state.last_analysis = analysis
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I looked at your screen time. "
                            f"{analysis['insight']} "
                            f"Try this: {analysis['suggestion']}"
                        ),
                    }
                )
                st.session_state.step = "dashboard"
                st.toast("Veriler başarıyla alındı ve analiz edildi! 🎉")
                st.rerun()

    with col2:
        if st.button("Skip for now ➡️", use_container_width=True):
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about your habits, or how to unplug…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    analysis = st.session_state.last_analysis
    if analysis:
        reply = analysis
    else:
        reply = (
            "You haven't uploaded screen time data yet. "
            "Set your city and hobbies in the sidebar to get offline event ideas."
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)


def render_disclaimer() -> None:
    """Ortadaki alanın en altına estetik bir sorumluluk reddi metni ekler."""
    st.markdown("---")
    st.caption(
        "⚠️ **Sorumluluk Reddi:** Bu uygulama yalnızca farkındalık ve dijital refah rehberliği "
        "amacıyla sunulmaktadır. Üretilen tavsiyeler yapay zeka tarafından oluşturulmuştur ve "
        "profesyonel tıbbi/psikolojik tavsiye niteliği taşımaz. Yüklenen veriler işlendikten sonra saklanmaz."
    )


def main() -> None:
    init_session_state()
    apply_custom_theme(st.session_state.theme)

    render_sidebar()

    # Sayfa Yönlendirme Mantığı (State-Based Routing)
    if st.session_state.step == "onboarding":
        render_onboarding_page()
    else:
        st.title("Digital Well-Being Dashboard")
        st.success("Veriler alındı! Koçunuz sorularınızı yanıtlamaya hazır.")
        render_analysis_card()
        render_chat()

    render_disclaimer()


if __name__ == "__main__":
    main()