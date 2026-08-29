# Digital well-being app

Streamlit app that helps you notice screen time and swap some of it for local, offline activities.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini API key to `.env` when you wire up real analysis (the current `ai_engine.py` returns mock data).

## Run

```bash
streamlit run app.py
```

## Project layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI: file upload, chat, session state |
| `ai_engine.py` | Placeholder analysis and event recommendations |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for API keys |

## Features (current)

- Upload a screen-time screenshot (`st.file_uploader`)
- Chat with a well-being coach (`st.chat_message` / `st.chat_input`)
- Persist chat and analysis in `st.session_state`
- Mock screen-time insights and local event ideas
