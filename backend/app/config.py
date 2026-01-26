from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure AI Search
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index_name: str = "docling-rag"

    @field_validator("azure_search_index_name")
    @classmethod
    def validate_index_name(cls, v: str) -> str:
        # Azure Search index names must be lowercase
        return v.lower()

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_model: str = "gpt-4o"
    azure_openai_embeddings: str = "text-embedding-3-large"
    azure_openai_embeddings_model: str = "text-embedding-3-large"

    # Azure OpenAI VLM (Vision Language Model) settings
    azure_openai_vlm_model: str = "gpt-4o"  # Vision-capable model for VLM pipeline
    azure_openai_vlm_max_tokens: int = 4096
    azure_openai_vlm_temperature: float = 0.0
    use_vlm_pipeline: bool = False  # Set to True to enable VLM pipeline for PDFs

    # App settings
    upload_dir: str = "/app/uploads"
    # 1536 for text-embedding-3-small, 3072 for text-embedding-3-large
    vector_dim: int = 3072

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
