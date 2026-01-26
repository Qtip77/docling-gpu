# Docling Backend

Document processing backend with Docling and RAG capabilities.

## Features

- Document parsing using Docling
- RAG (Retrieval Augmented Generation) agent with LangGraph
- Azure OpenAI integration
- Azure Cognitive Search integration

## Running

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
