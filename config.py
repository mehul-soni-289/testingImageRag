"""
Centralized configuration for the pipeline package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).parent.parent
    summaries_dir: Path = base_dir / "data" / "summaries"
    docstore_dir: Path = base_dir / "temp" / "pipeline_docstore"

    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_ANON_KEY", "")
    supabase_table: str = os.getenv("SUPABASE_TABLE", "page_summaries")
    supabase_query_fn: str = os.getenv("SUPABASE_QUERY_FN", "match_page_summaries")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")

    retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "6"))

    def validate_vector_store(self) -> None:
        if not self.supabase_url or not self.supabase_key:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in the environment."
            )
        if not self.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY must be set in the environment.")


settings = Settings()

