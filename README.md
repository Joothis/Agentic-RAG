# Agentic RAG Chatbot

A local agent-based retrieval-augmented generation (RAG) system that combines LLM capabilities with various tools for enhanced interaction.

![Architecture Diagram](./Agentic%20RAG.png)

## Features

- **Web Interface**: A user-friendly web interface for interacting with the chatbot.
- **Multi-Agent System**: A manager agent that coordinates specialized agents for retrieval, generation, and evaluation.
- **Iterative Refinement**: A critic agent that evaluates and provides feedback on generated responses.
- **Advanced Query Decomposition**: The manager agent decomposes complex queries into sub-queries for specialized agents.
- **RAG search using vector store (Chroma)**: With document loading and indexing.
- **SQL database querying**
- **Calculator for mathematical operations**
- **External API calling capabilities**: With robust parameter parsing and authentication.
- **Conversational memory for context retention**

## Setup

### Local Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. **Install Ollama and Download a Model**:
   - Download and install Ollama from [https://ollama.com/](https://ollama.com/) for your operating system.
   - After installation, open your terminal/command prompt and download a model. For example, to download Llama 2, run: `ollama run llama2` (You can also choose other models like `mistral` by running `ollama run mistral`).

4. Create the necessary data directories:

```bash
mkdir -p data/documents
```

### Docker Installation (Recommended)

**Prerequisites:**
- Docker and Docker Compose installed on your system
- Ollama running on your host machine with a model downloaded

1. **Start Ollama on your host machine:**

```bash
ollama serve
```

2. **Pull a model (e.g., Mistral):**

```bash
ollama pull mistral
```

3. **Build and run using Docker Compose:**

```bash
docker-compose up --build
```

This will build the image and start the container. The application will be available at `http://localhost:8000`.

4. **To stop the container:**

```bash
docker-compose down
```

**Alternative: Using Docker directly:**

```bash
# Build the image
docker build -t agentic-rag .

# Run the container (ensure Ollama is running on host)
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 agentic-rag
```

## Usage

### Local

Run the chatbot using:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Then, open your browser to `http://localhost:8000`.

### Docker

After running `docker-compose up --build`, open your browser to `http://localhost:8000`.

## Project Structure

```
.
├── data/
│   ├── documents/    # Store documents for RAG
│   └── database.db   # SQLite database
├── src/
│   ├── tools/
│   │   ├── rag_tool.py
│   │   ├── sql_tool.py
│   │   ├── calculator_tool.py
│   │   └── api_tool.py
│   ├── agent.py      # Main agent implementation
│   └── main.py       # FastAPI server
├── index.html        # Web interface
├── requirements.txt
├── Dockerfile
└── README.md
```

## Tools

1. **RAG Tool**: Vector similarity search through documents, with document loading and indexing.
2. **SQL Tool**: Execute queries on SQLite database.
3. **Calculator Tool**: Perform mathematical calculations.
4. **API Tool**: Make external API calls with robust parameter parsing and authentication.

## Contributing

Feel free to submit issues and enhancement requests!