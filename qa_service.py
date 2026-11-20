import hashlib
import json
from typing import AsyncGenerator

import redis.asyncio as redis
from openai import AsyncOpenAI

from config import settings
from retrieval import PolicyRetriever, RetrievedChunk

SYSTEM_PROMPT = """You are an enterprise policy assistant. Answer the employee's question using ONLY the provided policy excerpts.

Rules:
1. Cite every factual claim inline using [Policy ID, Section], e.g., [HR-402, Section 3.2].
2. State exact numbers, limits, approval roles, and exceptions verbatim where applicable.
3. If the excerpts do not contain enough information to answer confidently, respond: "I could not find an authoritative policy covering this question. Please contact the relevant policy owner." Do NOT guess or extrapolate.
4. Keep the answer concise, structured (use bullet points for conditions/steps), and professional."""


class PolicyQAService:
    def __init__(self, retriever: PolicyRetriever):
        self.retriever = retriever
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)

    async def _cache_key(self, query: str, user_groups: list[str]) -> str:
        corpus_ver = await self.redis.get("policy_corpus_version") or "1"
        normalized_q = " ".join(query.lower().strip().split())
        groups_sig = ",".join(sorted(set(user_groups)))
        raw = f"{corpus_ver}:{groups_sig}:{normalized_q}"
        return f"qa_cache:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def invalidate_cache_on_ingest(self):
        await self.redis.incr("policy_corpus_version")

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, c in enumerate(chunks, 1):
            blocks.append(
                f"--- Excerpt {i} (Policy: {c.policy_id} | Title: {c.title} | "
                f"Section: {c.section} | Effective: {c.effective_date}) ---\n{c.text}"
            )
        return "\n\n".join(blocks)

    async def answer_stream(
        self, query: str, user_groups: list[str]
    ) -> AsyncGenerator[str, None]:
        cache_key = await self._cache_key(query, user_groups)
        cached = await self.redis.get(cache_key)
        if cached:
            payload = json.loads(cached)
            yield json.dumps({"type": "sources", "sources": payload["sources"], "cached": True}) + "\n"
            yield json.dumps({"type": "token", "content": payload["answer"]}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        chunks = await self.retriever.retrieve(query, user_groups)
        if not chunks:
            msg = "I could not find an active policy you have access to that answers this question."
            yield json.dumps({"type": "sources", "sources": [], "cached": False}) + "\n"
            yield json.dumps({"type": "token", "content": msg}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        sources = [
            {
                "policy_id": c.policy_id,
                "title": c.title,
                "section": c.section,
                "effective_date": c.effective_date,
                "source_uri": c.source_uri,
                "relevance_score": round(c.score, 3),
            }
            for c in chunks
        ]
        yield json.dumps({"type": "sources", "sources": sources, "cached": False}) + "\n"

        context_str = self._format_context(chunks)
        stream = await self.openai.chat.completions.create(
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=700,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Policy Excerpts:\n{context_str}\n\nQuestion: {query}"},
            ],
        )

        full_answer = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_answer.append(delta)
                yield json.dumps({"type": "token", "content": delta}) + "\n"

        # Populate cache
        await self.redis.setex(
            cache_key,
            settings.cache_ttl_seconds,
            json.dumps({"answer": "".join(full_answer), "sources": sources}),
        )
        yield json.dumps({"type": "done"}) + "\n"