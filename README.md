# 🏆 SportGPT - Intelligent Sports Assistant

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered sports expert system with Retrieval-Augmented Generation (RAG) capabilities, providing professional insights into sports rules, training plans, and real-time match analysis.

## ✨ Features

### Core Capabilities
- **Document Intelligence**  
  ![RAG Architecture](docs/rag_architecture.png)  
  Ingests PDF/TXT documents from `data/knowledge_base/` using hybrid retrieval:
  - Semantic search with FAISS vectors
  - Keyword matching with BM25

- **Expert-Level Analysis**
  - Real-time match statistics integration
  - Rule interpretation with official citations
  - Personalized training plan generation

- **Multi-Modal Support**
  - Text explanations with markdown formatting
  - Tactical diagrams (SVG)
  - Video highlights integration

### Technical Highlights
- 🚀 **Hybrid Retrieval** combining dense & sparse vectors
- 🔐 **Secure API** key management with environment variables
- 📈 **Performance** optimized with FAISS indexing
- 🔄 **Auto-Reload** knowledge base on document changes

## 🛠️ Tech Stack

| Component              | Technology                          |
|------------------------|-------------------------------------|
| Language Model         | OpenAI GPT-4                        |
| Embeddings             | OpenAI text-embedding-3-small       |
| Vector Database        | FAISS                               |
| Document Loaders       | PyPDF, TextLoader                   |
| Frontend               | Streamlit                           |
| Deployment             | Docker + AWS EC2                    |

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API key
- 2GB+ free memory

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/sports-chatbot.git
cd sports-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
