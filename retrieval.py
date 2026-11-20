import asyncio
import re
from dataclasses import dataclass

from flashrank import Ranker, RerankRequest
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models

from config import settings
from ingestion import encode_bm25_sparse


@dataclass
class RetrievedChunk:
    policy_id: str
    title: str
    section: str
    text: str
    source_uri: str
    effective_date: str
    score: float


class PolicyRetriever:
    def __init__(self):
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        try:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        except Exception as e:
            print(f"[WARN] FlashRank download failed ({e}); using local RRF + lexical fallback reranker.")
            self.reranker = None

    async def _embed_dense(self, query: str) -> list[float]:
        resp = await self.openai.embeddings.create(model=settings.embedding_model, input=query)
        return resp.data[0].embedding

    async def _embed_sparse(self, query: str) -> models.SparseVector:
        indices, values = encode_bm25_sparse(query)
        return models.SparseVector(indices=indices, values=values)

    @staticmethod
    def _fallback_rerank(query: str, passages: list[dict]) -> list[dict]:
        q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        total = max(len(passages), 1)
        for rank, p in enumerate(passages):
            p_terms = set(re.findall(r"[a-z0-9]+", p["text"].lower()))
            overlap = len(q_terms & p_terms) / max(len(q_terms), 1)
            rrf_prior = 1.0 - (rank / total) * 0.5
            score = 0.6 * rrf_prior + 0.4 * overlap
            scored.append({**p, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def retrieve(self, query: str, user_groups: list[str]) -> list[RetrievedChunk]:
        dense_vec, sparse_vec = await asyncio.gather(
            self._embed_dense(query),
            self._embed_sparse(query),
        )

        rbac_filter = models.Filter(
            must=[
                models.FieldCondition(key="status", match=models.MatchValue(value="active")),
                models.FieldCondition(key="allowed_groups", match=models.MatchAny(any=user_groups)),
            ]
        )

        search_res = await self.qdrant.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=settings.prefetch_limit, filter=rbac_filter),
                models.Prefetch(query=sparse_vec, using="sparse", limit=settings.prefetch_limit, filter=rbac_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
            with_payload=True,
        )

        if not search_res.points:
            return []

        passages = [
            {"id": idx, "text": pt.payload["text"], "meta": pt.payload}
            for idx, pt in enumerate(search_res.points)
        ]

        if self.reranker is not None:
            reranked = await asyncio.to_thread(
                self.reranker.rerank,
                RerankRequest(query=query, passages=passages),
            )
        else:
            reranked = self._fallback_rerank(query, passages)

        results: list[RetrievedChunk] = []
        for item in reranked[: settings.rerank_top_k]:
            if item["score"] < settings.min_rerank_score:
                continue
            m = item["meta"]
            results.append(
                RetrievedChunk(
                    policy_id=m["policy_id"],
                    title=m["title"],
                    section=m["section"],
                    text=m["text"],
                    source_uri=m["source_uri"],
                    effective_date=m["effective_date"],
                    score=float(item["score"]),
                )
            )
        return results