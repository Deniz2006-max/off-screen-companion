# 📱 Off-Screen Companion: AI-Powered Screen Time Detox

An intelligent digital well-being assistant built with **LangChain**, **LangGraph**, **Gemini 2.5/3.6 Flash Vision**, and **Tavily AI**. The application analyzes your mobile screen time screenshots via multimodal OCR, detects habits, and recommends real-time local events or tailored offline hobbies to swap your screen time.

---

## 🚀 Key Features

- **Multimodal Screen Time OCR:** Upload iOS/Android screen time screenshots for automatic habit extraction using Gemini Vision.
- **Deterministic Guardrails & Rejection Handling:** Out-of-scope images (e.g., random memes, landscape photos) or invalid text prompts are instantly flagged and rejected with helpful user feedback.
- **Intent-Based Routing:** Smart flow routing that distinguishes between screen-time analysis requests and offline activity searches.
- **Real-Time Event Grounding:** Tavily API integration to fetch live local concerts, exhibitions, and workshops based on your location.

---



## 📐 System Architecture & Agent Flow

System Architecture

---



## 🛠️ Technology Stack & AI Reasoning



### 1. LangChain & LangGraph (Agent Orchestration & Core Infrastructure)

- **Why it was chosen:** LangChain provides the foundational abstractions for model bindings, prompt templates, and tool integrations, while LangGraph supplies state-machine control for cyclical multi-agent routing.
- **Role in Project:** Manages state persistence `st.session_state` integration), node execution order, dynamic context passing, and tool invocations across the agent pipeline.



### 2. Scope Guard Router & Rejection Node (Input Validation & Guardrails)

- **Why it was chosen:** To protect API quotas and prevent the model from processing invalid inputs, irrelevant images (non-screen-time photos), or off-topic prompts.
- **Role in Project:** Evaluates incoming user inputs before heavy analysis. If an invalid prompt or unrelated screenshot is detected, the flow immediately routes to the **Rejection Node**, delivering a friendly corrective message to the user without executing unnecessary downstream tasks.



### 3. Tavily AI (Real-Time Local Event Grounding)

- **Why it was chosen:** Standard LLMs rely on static training data and risk hallucinating venue names, dates, or ticket details for public events.
- **Role in Project:** Queries live search APIs based on user location to retrieve verified real-time event data (concerts, theater, exhibitions, open-air cinema) and supplies it directly into the generation pipeline.



### 4. Google Gemini 3.6 Flash (Multimodal OCR & Reasoning)

- **Why it was chosen:** High-throughput vision capability required for accurate Image-to-Text extraction from mobile dashboards.
- **Role in Project:** Performs zero-shot OCR extraction on uploaded screenshots to identify precise active hours and top app durations without manual user input.

---



## 💻 Quick Start



### 1. Installation

```bash

python -m venv .venv

source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

