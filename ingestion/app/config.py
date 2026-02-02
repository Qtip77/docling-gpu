"""Ingestion service configuration."""
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis
    redis_url: str = "redis://redis:6379/1"
    
    # Upload directory
    upload_dir: str = "/app/uploads"
    
    # Mistral Azure MaaS
    mistral_azure_endpoint: str = ""
    mistral_azure_api_key: str = ""
    mistral_azure_model: str = "mistral-document-ai-2505"

    # OCR Processing
    ocr_page_batch_size: int = 10
    ocr_max_concurrent: int = 5
    ocr_timeout: int = 120
    ocr_table_format: str = "markdown"
    ocr_max_pages_per_request: int = 30  # Mistral Document AI limit

    # Document Processing
    max_concurrent_docs: int = 5

    # Chunking Strategy
    # "hierarchical" = fast rule-based (~10ms), good for general docs
    # "agentic" = LLM-powered (~2-5s/page), best for contracts/legal/ballots
    chunking_strategy: Literal["hierarchical", "agentic"] = "hierarchical"
    
    # Agentic chunking config (only used if chunking_strategy="agentic")
    agentic_chunk_size: int = 1024  # Target tokens per chunk
    agentic_model: str = "gpt-4.1-mini"  # Model for chunking decisions
    agentic_describe_images: bool = True  # Use vision model to describe images
    agentic_vision_model: str = "gpt-4o"  # Vision model for image descriptions

    # Embedding
    embedding_batch_size: int = 500
    embedding_max_concurrent: int = 4

    # Azure AI Search
    azure_search_endpoint: str = ""
    azure_search_key: str = ""
    azure_search_index_name: str = "docling-rag"
    search_upload_batch_size: int = 1000
    search_max_concurrent: int = 3

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embeddings: str = "text-embedding-3-large"
    azure_openai_embeddings_model: str = "text-embedding-3-large"

    # Vector dimensions
    vector_dim: int = 3072

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
