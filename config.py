from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Vector DB & Cache
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_policies"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # Models (swap openai_base_url for local vLLM/TGI to cut latency & keep data on-prem)
    openai_api_key: str = "your-api-key"
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_model: str = "gpt-4o-mini"

    # Retrieval tuning
    chunk_max_tokens: int = 450
    chunk_overlap_tokens: int = 60
    prefetch_limit: int = 30
    rerank_top_k: int = 5
    min_rerank_score: float = 0.25  # Abstain if no chunk clears threshold

    class Config:
        env_file = ".env"


settings = Settings()