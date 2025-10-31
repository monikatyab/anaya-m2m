# Anaya AI - Emotional Wellness Assistant

An intelligent, memory-enabled AI companion for emotional wellness support.

## Overview

Anaya AI combines multiple AI agents, retrieval-augmented generation (RAG), and persistent memory to provide personalized therapeutic conversations. Adaptable for any user group requiring emotional support.

**Key Features:**
- Multi-agent therapeutic system with crisis detection
- Knowledge grounding using your own documents (RAG)
- Persistent memory across sessions
- Customizable for different communities

---

## Features

**Intelligent Conversation Flow**
- Natural therapeutic progression through understanding, exploring, coping, and integration
- Context-aware responses based on history
- Turn-by-turn intent tracking

**Knowledge Base Integration**
- Upload wellness documents (.docx, .pdf, .txt)
- Automatic indexing and semantic search
- Evidence-based responses from your materials

**Dual Memory System**
- Short-Term Memory: Tracks conversation state within sessions
- Long-Term Memory: Learns patterns and progress over time

**Crisis Management**
- Real-time detection of crisis keywords
- Immediate supportive responses
- Resource referrals

**Multi-User Support**
- Personalized profiles
- Individual tracking
- Unlimited scalability

**Dual Interfaces**
- Command line: `python main.py`
- Web UI: `streamlit run streamlit_app.py`

---

## Quick Start

**Prerequisites:**
- Python 3.12+
- Google AI API key

**Installation:**

```bash
# Clone repository
git clone https://github.com/yourusername/anaya-ai.git
cd anaya-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env_example .env
# Edit .env and add your GOOGLE_API_KEY

# Add documents to rag/knowledge_base/

# Process knowledge base
python rag/ingest_documents.py

# Run application
python main.py
# or: streamlit run streamlit_app.py
```

---

## Project Structure

```
anaya-ai/
├── agents/           # AI agent modules
├── core/             # System logic (workflow, memory)
├── rag/              # Knowledge base and RAG system
│   ├── knowledge_base/     # Your documents here
│   └── anaya_knowledge_db/ # Vector database
├── data/             # User profiles and logs
├── main.py           # Command-line interface
└── streamlit_app.py  # Web interface
```

---

## Architecture

**Agent Flow:**
User Input → Short-Term Memory → Crisis Check → Planner → Execute (Wellness + RAG, Reflection, Factual) → Synthesis → Dialogue Manager → Response → Memory Logging

**Memory:**
- STM logs every turn with emotions and context
- LTM analyzes sessions for insights and progress

**RAG Pipeline:**
Documents → Chunking → Embeddings → Vector DB → Search → Context for agents

---

## Customization

**Adapt to different user groups (no coding):**
1. Edit `data/User_Data.csv` - Update user profiles
2. Edit `agents/wellness_assistant_agent.py` line 30 - Change persona description
3. Replace documents in `rag/knowledge_base/` - Add domain-specific content
4. Run `python rag/ingest_documents.py`

**Modify RAG settings in `rag/simple_rag.py`:**
```python
CHUNK_SIZE = 500      # Increase for more context per result
CHUNK_OVERLAP = 100   # Increase to preserve continuity
TOP_K_RESULTS = 6     # Increase for more comprehensive answers
```

---

## Testing

```bash
python main.py
# Enter User ID: U0001
# Test with: "I'm feeling anxious", "I don't know what to do"
```

Check logs in `data/STM_Data.csv` and `data/LTM_Data.csv`

---

## Data Privacy

All data stored locally in CSV files. No external transmission except Google AI API for inference. User data never leaves your machine.

---

## Troubleshooting

**"No vector database found"** → Run `python rag/ingest_documents.py`

**"Invalid API key"** → Check `.env` file has correct `GOOGLE_API_KEY`

**"Module not found"** → Activate venv and run `pip install -r requirements.txt`

**Agents not using documents** → Verify files in `rag/knowledge_base/` and ingestion completed

---

## Tech Stack

Built with LangChain, LangGraph, Google Gemini, ChromaDB, and Streamlit.

## License

MIT License
