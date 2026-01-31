"""FastAPI application with lifespan for worker startup."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import documents
from app.workers import processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start background workers
    await processor.start_workers()
    yield
    # Shutdown: stop workers
    await processor.stop_workers()


app = FastAPI(
    title="Document Ingestion API",
    description="High-performance document ingestion with Mistral OCR",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Document Ingestion API", "docs": "/docs"}
