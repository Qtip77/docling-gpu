# Docling RAG

Full-stack RAG application using Docling for document parsing, Azure AI Search for vector storage, and Azure OpenAI for embeddings and chat.

## Features

- **Document Upload**: Upload PDFs with tables, images, and handwritten notes
- **OCR Processing**: EasyOCR for handwritten content recognition
- **Table Extraction**: TableFormer ACCURATE mode for complex tables
- **Vector Search**: Hybrid search using Azure AI Search
- **RAG Chat**: Retrieval-augmented generation with Azure OpenAI GPT-4o

## Quick Start

1. Copy `env.example` to `.env` and fill in your Azure credentials:

```bash
cp env.example .env
```

2. Run with Docker Compose:

```bash
docker compose up --build
```

3. Access the application:
   - Frontend: http://localhost:3000
   - API docs: http://localhost:8000/docs

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint URL |
| `AZURE_SEARCH_KEY` | Azure AI Search admin key |
| `AZURE_SEARCH_INDEX_NAME` | Index name (default: docling-rag) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | API version (default: 2024-10-21) |
| `AZURE_OPENAI_CHAT_MODEL` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDINGS` | Embeddings model deployment name |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React UI      │────▶│   FastAPI       │────▶│  Azure AI       │
│   (Port 3000)   │     │   (Port 8000)   │     │  Search         │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Azure OpenAI   │
                        │  (Embeddings +  │
                        │   Chat)         │
                        └─────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/documents/upload` | POST | Upload and process document |
| `/api/documents` | GET | List processed documents |
| `/api/documents/{id}` | DELETE | Delete document and chunks |
| `/api/documents/status/{job_id}` | GET | Get processing status |
| `/api/search` | POST | Vector search |
| `/api/chat` | POST | RAG chat |
| `/health` | GET | Health check |

## Development

### Backend only

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend only

```bash
cd frontend
npm install
npm run dev
```

## Notes

- CPU-only Docling configuration (no GPU required)
- Processing may be slower for large documents
- No authentication - all endpoints are public
