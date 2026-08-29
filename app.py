"""Streamlit entry point for the digital well-being app."""

from __future__ import annotations

import streamlit as st

from ai_engine import analyze_screen_time, get_event_recommendations

st.set_page_config(
    page_title="Digital Well-being",
    page_icon="🌿",
    layout="centered",
)


def init_session_state() -> None:
    defaults: dict[str, object] = {
        "messages": [],
        "last_analysis": None,
        "last_events": None,
        "city": "",
        "hobbies": "",
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

        if st.button("Analyze screen time", type="primary"):
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

        if st.button("Suggest local events"):
            events = get_event_recommendations(
                st.session_state.city,
                st.session_state.hobbies,
            )
            st.session_state.last_events = events
            lines = ["Here are some offline ideas near you:"]
            for event in events:
                lines.append(
                    f"- **{event['title']}** ({event['when']}, {event['city']}) — {event['why']}"
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": "\n".join(lines)}
            )


def render_analysis_card() -> None:
    analysis = st.session_state.last_analysis
    if not analysis:
        return

    st.subheader("Latest screen-time snapshot")
    cols = st.columns(2)
    cols[0].metric("Estimated hours", f"{analysis['total_hours']}")
    cols[1].metric("Top app", analysis["top_apps"][0]["name"])
    st.caption(analysis["insight"])


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
        reply = (
            f"{analysis['insight']} "
            "If you want something local, set your city and hobbies in the sidebar."
        )
    else:
        reply = (
            "Upload a screen-time screenshot or add a note, then tap "
            "**Analyze screen time**. I can also suggest events once you set a city."
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)


def main() -> None:
    init_session_state()
    st.title("Digital well-being")
    st.write(
        "Upload a screen-time screenshot, chat about your habits, "
        "and get offline event ideas."
    )
    render_sidebar()
    render_analysis_card()
    render_chat()


if __name__ == "__main__":
    main()
