# Agentic RAG Chatbot

A local agent-based retrieval-augmented generation (RAG) system that combines LLM capabilities with various tools for enhanced interaction.

## Features

- RAG search using vector store (Chroma)
- SQL database querying
- Calculator for mathematical operations
- External API calling capabilities
- Conversational memory for context retention

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

4. Create the necessary data directories:

```bash
mkdir -p data/documents
```

## Usage

Run the chatbot using:

```bash
python src/main.py
```

Optional parameters:

- `--model-name`: Specify the LLM model (default: "gpt-3.5-turbo")
- `--temperature`: Set response temperature (default: 0.7)
- `--documents-path`: Set path to RAG documents (default: "./data/documents")
- `--db-path`: Set path to SQLite database (default: "./data/database.db")

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
│   └── main.py       # CLI interface
├── requirements.txt
└── README.md
```

## Tools

1. **RAG Tool**: Vector similarity search through documents
2. **SQL Tool**: Execute queries on SQLite database
3. **Calculator Tool**: Perform mathematical calculations
4. **API Tool**: Make external API calls

## Contributing

Feel free to submit issues and enhancement requests!
