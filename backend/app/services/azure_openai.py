from openai import AzureOpenAI
from app.config import settings


_client: AzureOpenAI | None = None


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    return _client


def embed_text(text: str) -> list[float]:
    """Generate embedding vector for text using Azure OpenAI."""
    client = get_client()
    response = client.embeddings.create(
        input=text,
        model=settings.azure_openai_embeddings
    )
    return response.data[0].embedding


def generate_chat_response(prompt: str, system_message: str | None = None) -> str:
    """Generate chat completion using Azure OpenAI."""
    client = get_client()
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    completion = client.chat.completions.create(
        model=settings.azure_openai_chat_model,
        messages=messages,
        temperature=0.7
    )
    return completion.choices[0].message.content


def generate_rag_response(query: str, context_chunks: list[str]) -> str:
    """Generate RAG response with retrieved context."""
    context_str = "\n---\n".join(context_chunks)
    
    prompt = f"""You are an AI assistant answering questions based on provided documents.
Use ONLY the context below to answer. If the answer isn't in the context, say you don't know.

Context:
{context_str}

Question: {query}
Answer:"""
    
    return generate_chat_response(prompt)
