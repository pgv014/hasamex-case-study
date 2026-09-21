# Hasamex Expert Call Transcript Analyzer

A lightweight, AI-powered application designed to analyze expert call transcripts, extract actionable insights, and enable cross-document Q&A with strict adherence to factual accuracy (exact quotes and timestamps).

## 🚀 Features
1. **Interview Guide Analysis**: Automatically maps interview guide questions to synthesized answers backed by exact transcript quotes and timestamps.
2. **Themes & Disagreements**: Performs a global analysis across all transcripts to identify common market themes and conflicting expert viewpoints.
3. **Interactive Q&A**: A RAG-powered chat interface allowing users to ask ad-hoc questions across all transcripts, with guaranteed citation of sources and timestamps.

## 🏗️ Architecture & Key Decisions
- **Hybrid Context Strategy**: Uses RAG (FAISS + OpenAI Embeddings) for scalable, targeted Q&A, but passes the *full text* to the LLM for the "Themes & Disagreements" analysis. This prevents chunking from severing cross-call contextual nuances.
- **Accuracy-First Prompting**: All prompts use `temperature=0` and strict negative/positive constraints forcing the LLM to output exact quotes and timestamps, minimizing hallucination.
- **Tech Stack**: Streamlit (UI), LangChain (Orchestration), OpenAI GPT-4o (Reasoning), FAISS (Vector Search).

## 🛠️ How to Run Locally
1. Clone this repository and navigate to the folder.
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `data/` folder and place the 4 case pack files inside (`Interview_Guide.txt`, `Transcript_1_France.txt`, `Transcript_2_Germany.txt`, `Transcript_3_UK.txt`).
5. Run the app: `streamlit run app.py`
6. Enter your OpenAI API key in the sidebar to begin.

## 📈 Product Thinking & Next Steps
While this MVP operates on pre-transcribed text with timestamps, a production V2 would ingest raw audio, utilize Whisper for transcription + speaker diarization, and auto-generate timestamps, creating a fully end-to-end "audio-to-insights" pipeline.