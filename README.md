# 🤖 Agentic RAG System

An advanced AI-powered research assistant that combines **Retrieval-Augmented Generation (RAG)** with **Model Context Protocol (MCP)** integration. Powered by **OpenRouter's free LLM models** (DeepSeek V3), this system provides intelligent document analysis with meaningful insights.

![Architecture Diagram](./Agentic%20RAG.png)

## ✨ Key Features

### 🧠 AI-Powered Analysis
- **OpenRouter Integration**: Access to top free models (DeepSeek V3, Llama 3.3 70B)
- **Multi-Agent Architecture**: Manager, Retrieval, Generative, and Critic agents
- **Quality Assurance**: Built-in response evaluation and refinement

### 📊 Meaningful Insights
- **Usage Analytics**: Track query patterns and response quality
- **Topic Analysis**: Automatic identification of discussion themes
- **Trend Detection**: Identify patterns in user queries
- **Recommendations**: AI-generated suggestions for knowledge base improvements

### 🔧 MCP (Model Context Protocol)
- **Standardized Tool Interface**: JSON-RPC based tool orchestration
- **Resource Management**: Dynamic resource discovery and access
- **Prompt Templates**: Pre-configured prompts for common tasks

### 🛠️ Available Tools
1. **RAG Search**: Vector similarity search through documents
2. **SQL Query**: Execute queries on SQLite databases
3. **Calculator**: Mathematical operations and expressions
4. **API Caller**: HTTP requests to external APIs
5. **Insights Analyzer**: Statistical and trend analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenRouter API Key (free tier available)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repo-url>
cd Agentic-RAG
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
# .env file is pre-configured with your API key
# Optionally modify MODEL_NAME for different models:
# - deepseek/deepseek-chat-v3-0324:free (Best reasoning)
# - meta-llama/llama-3.3-70b-instruct:free (Great balance)
# - google/gemma-2-9b-it:free (Fast responses)
```

3. **Create data directories:**
```bash
mkdir -p data/documents data/vector_store
```

4. **Run the application:**
```bash
cd src
python main.py
```

5. **Open your browser:**
```
http://localhost:8000
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/chat` | POST | Send messages to AI |
| `/insights` | GET | Get usage analytics |
| `/tools` | GET | List available tools |
| `/mcp` | POST | MCP protocol endpoint |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation |

### Example: Chat Request
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze the trends in my data", "with_critique": true}'
```

### Example: MCP Tool Call
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/list",
    "id": 1,
    "params": {}
  }'
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface                            │
│              (Real-time Insights Dashboard)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  FastAPI Server                             │
│         (/chat, /insights, /mcp, /tools)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  AgenticRAG Core                            │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │ OpenRouter  │ │ MCP Server   │ │ Insight Generator    │  │
│  │ LLM Client  │ │ (Tools/Res)  │ │ (Analytics)          │  │
│  └─────────────┘ └──────────────┘ └──────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    ReAct Agent                          │ │
│  │    (Reasoning + Acting with Tool Orchestration)         │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   Critic Agent                          │ │
│  │        (Quality Evaluation & Refinement)                │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                      Tools Layer                            │
│  ┌──────┐ ┌──────┐ ┌──────────┐ ┌─────┐ ┌────────────────┐  │
│  │ RAG  │ │ SQL  │ │Calculator│ │ API │ │ Insights Tool  │  │
│  └──────┘ └──────┘ └──────────┘ └─────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Agentic-RAG/
├── src/
│   ├── agent.py          # Main AgenticRAG with OpenRouter
│   ├── main.py           # FastAPI server with MCP
│   ├── mcp_server.py     # MCP protocol implementation
│   └── tools/
│       ├── __init__.py
│       ├── rag_tool.py       # Document search
│       ├── sql_tool.py       # Database queries
│       ├── calculator_tool.py # Math operations
│       ├── api_tool.py       # HTTP requests
│       └── insights_tool.py  # Data analytics
├── data/
│   ├── documents/        # Documents for RAG
│   ├── vector_store/     # ChromaDB embeddings
│   └── database.db       # SQLite database
├── .env                  # API configuration
├── index.html            # Web interface
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🎯 Why This Project?

### The Problem
Traditional chatbots and RAG systems often provide:
- Raw data without context or actionable insights
- No quality assurance on generated responses
- Limited tool integration capabilities
- No analytics on usage patterns

### The Solution
This Agentic RAG system addresses these issues by:

1. **Generating Meaningful Insights**: Not just answering questions, but analyzing patterns, detecting trends, and providing recommendations
2. **Quality Assurance**: Built-in critic agent evaluates every response for accuracy, relevance, and completeness
3. **MCP Integration**: Standardized protocol for tool orchestration, making it extensible and interoperable
4. **Free & Accessible**: Powered by OpenRouter's free tier, making advanced AI accessible to everyone

### Use Cases
- **Research Assistance**: Analyze documents and extract key insights
- **Data Analysis**: Statistical analysis with trend detection and recommendations
- **Knowledge Management**: Build and query your own knowledge base
- **API Orchestration**: Coordinate multiple data sources through a single interface

## 🔐 Security Note

The `.env` file contains your OpenRouter API key. Make sure to:
- Never commit `.env` to public repositories
- Rotate your API key if exposed
- Use environment variables in production

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.