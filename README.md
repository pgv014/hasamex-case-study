# Hasamex Expert Call Transcript Analyzer

A lightweight, 100% local, privacy-first AI application designed to analyze expert call transcripts, extract actionable insights, and enable cross-document Q&A with strict adherence to factual accuracy (exact quotes and timestamps).

## 🚀 Features
1. **Interview Guide Analysis**: Automatically maps interview guide questions to synthesized answers backed by exact transcript quotes and timestamps.
2. **Themes & Disagreements**: Performs a global analysis across all transcripts to identify common market themes and conflicting expert viewpoints.
3. **Interactive Q&A**: A RAG-powered chat interface allowing users to ask ad-hoc questions across all transcripts, with guaranteed citation of sources and timestamps.

## 🏗️ Architecture & Key Decisions
- **100% Local & Private (Ollama)**: Built using Ollama (Llama 3.2) and local embeddings (`nomic-embed-text`). This eliminates ongoing API costs, guarantees zero data privacy risks for sensitive expert call transcripts, and proves a production-ready architecture for enterprise environments.
- **Hybrid Context Strategy**: Uses local FAISS vector search for scalable, targeted Q&A, but passes the *full text* to the LLM for the "Themes & Disagreements" analysis. This prevents chunking from severing cross-call contextual nuances.
- **Modern LCEL Pipeline**: Utilizes LangChain's modern Expression Language (LCEL) for a clean, maintainable, and highly performant RAG pipeline.
- **Accuracy-First Prompting**: All prompts use `temperature=0` and strict negative/positive constraints forcing the LLM to output exact quotes and timestamps, heavily mitigating hallucination risks.

## 🛠️ How to Run Locally
1. Clone this repository: `git clone https://github.com/YOUR_USERNAME/hasamex-case-study.git`
2. Install Ollama from [ollama.com](https://ollama.com) and start the service.
3. Pull the required local models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text