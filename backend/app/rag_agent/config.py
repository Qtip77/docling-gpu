"""Configuration for Azure OpenAI models used in LangGraph agents."""
import os
from typing import Optional
from langchain_openai import AzureChatOpenAI


def _get_temperature(env_var: str, default: float) -> Optional[float]:
    """
    Get temperature from environment variable.
    
    Returns None if set to empty string or 'default' (for models that
    only support default temperature like gpt-5-nano).
    
    Args:
        env_var: Environment variable name
        default: Default temperature value if not set
        
    Returns:
        Temperature float or None to use model default
    """
    value = os.getenv(env_var, "").strip().lower()
    if value in ("", "default", "none"):
        return None
    try:
        return float(value)
    except ValueError:
        return default


def get_analyst_llm() -> AzureChatOpenAI:
    """
    Get configured Azure OpenAI LLM for analyst agents (smaller model).
    
    Uses configured analyst deployment for cost-effective parallel processing.
    Optimized for fast, focused chunk analysis.
    
    Environment variables:
        AZURE_OPENAI_ANALYST_DEPLOYMENT: Model deployment name
        AZURE_OPENAI_ANALYST_TEMPERATURE: Temperature (0-2), or 'default' for model default
    """
    temperature = _get_temperature("AZURE_OPENAI_ANALYST_TEMPERATURE", 0.1)
    
    kwargs = {
        "azure_deployment": os.getenv("AZURE_OPENAI_ANALYST_DEPLOYMENT", "gpt-4o-mini"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        "max_retries": 2,
    }
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    return AzureChatOpenAI(**kwargs)


def get_orchestrator_llm() -> AzureChatOpenAI:
    """
    Get configured Azure OpenAI LLM for orchestrator (larger model).
    
    Uses configured orchestrator deployment for high-quality synthesis
    and citation generation.
    
    Environment variables:
        AZURE_OPENAI_ORCHESTRATOR_DEPLOYMENT: Model deployment name
        AZURE_OPENAI_ORCHESTRATOR_TEMPERATURE: Temperature (0-2), or 'default' for model default
    """
    temperature = _get_temperature("AZURE_OPENAI_ORCHESTRATOR_TEMPERATURE", 0.2)
    
    kwargs = {
        "azure_deployment": os.getenv("AZURE_OPENAI_ORCHESTRATOR_DEPLOYMENT", "gpt-4o"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        "max_retries": 2,
    }
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    return AzureChatOpenAI(**kwargs)
