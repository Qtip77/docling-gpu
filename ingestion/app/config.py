from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Mistral Azure MaaS
    mistral_azure_endpoint: str
    mistral_azure_api_key: str

    # OCR Processing
    ocr_page_batch_size: int = 10
    ocr_max_concurrent: int = 5
    ocr_timeout: int = 120
    ocr_table_format: str = "html"

    # Document Processing
    max_concurrent_docs: int = 5

    # Embedding
    embedding_batch_size: int = 500
    embedding_max_concurrent: int = 4

    # Azure AI Search
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index_name: str = "docling-rag"
    search_upload_batch_size: int = 1000
    search_max_concurrent: int = 3

    @field_validator("azure_search_index_name")
    @classmethod
    def validate_index_name(cls, v: str) -> str:
        return v.lower()

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embeddings: str = "text-embedding-3-large"
    azure_openai_embeddings_model: str = "text-embedding-3-large"

    # Vector dimensions
    vector_dim: int = 3072

    # Redis
    redis_url: str = "redis://langgraph-redis:6379/1"

    # App
    upload_dir: str = "/app/uploads"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
