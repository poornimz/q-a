import hashlib
import re
import uuid
from collections import Counter
from dataclasses import dataclass

import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient, models

from config import settings

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that",
    "the", "their", "then", "there", "these", "they", "this", "to", "was",
    "will", "with",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def encode_bm25_sparse(text: str, k1: float = 1.2) -> tuple[list[int], list[float]]:
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]
    if not tokens:
        return [0], [0.0]
    counts = Counter(tokens)
    seen: dict[int, float] = {}
    for tok, tf in counts.items():
        idx = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest(), "little") & 0x7FFFFFFF
        weight = (tf * (k1 + 1.0)) / (tf + k1)
        seen[idx] = seen.get(idx, 0.0) + weight
    sorted_items = sorted(seen.items())
    return [i for i, _ in sorted_items], [float(v) for _, v in sorted_items]


@dataclass
class PolicyMetadata:
    policy_id: str
    title: str
    department: str
    effective_date: str
    status: str
    allowed_groups: list[str]
    source_uri: str


class PolicyChunker:
    """Splits policy documents along section boundaries while preserving hierarchy."""

    HEADING_RE = re.compile(r"^(#{1,4}\s+.+|\d+(?:\.\d+)*\s+[A-Z][^\n]{2,80})$", re.MULTILINE)

    def __init__(self, max_tokens: int = settings.chunk_max_tokens):
        self.max_tokens = max_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in doc)

    def chunk_document(self, text: str, meta: PolicyMetadata) -> list[dict]:
        matches = list(self.HEADING_RE.finditer(text))
        sections: list[tuple[str, str]] = []

        if not matches:
            sections = [("General", text.strip())]
        else:
            if matches[0].start() > 0:
                preamble = text[: matches[0].start()].strip()
                if preamble:
                    sections.append(("Overview", preamble))
            for i, m in enumerate(matches):
                heading = m.group(0).lstrip("# ").strip()
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[start:end].strip()
                if body:
                    sections.append((heading, body))

        chunks = []
        for section_title, body in sections:
            prefix = f"[Policy: {meta.policy_id} | {meta.title} | Section: {section_title}]"
            tokens = self.enc.encode(body)

            if len(tokens) <= self.max_tokens:
                chunks.append(self._build_chunk(prefix, body, section_title, meta))
            else:
                step = self.max_tokens - settings.chunk_overlap_tokens
                for start_idx in range(0, len(tokens), step):
                    sub_tokens = tokens[start_idx : start_idx + self.max_tokens]
                    sub_text = self.enc.decode(sub_tokens)
                    chunks.append(self._build_chunk(prefix, sub_text, section_title, meta))
        return chunks

    @staticmethod
    def _build_chunk(prefix: str, body: str, section: str, meta: PolicyMetadata) -> dict:
        full_text = f"{prefix}\n{body}"
        chunk_hash = hashlib.sha256(f"{meta.policy_id}:{section}:{body}".encode()).hexdigest()
        return {
            "id": str(uuid.UUID(chunk_hash[:32])),
            "text": full_text,
            "payload": {
                "text": full_text,
                "policy_id": meta.policy_id,
                "title": meta.title,
                "section": section,
                "department": meta.department,
                "effective_date": meta.effective_date,
                "status": meta.status,
                "allowed_groups": meta.allowed_groups,
                "source_uri": meta.source_uri,
            },
        }


class PolicyIndexer:
    def __init__(self):
        self.qdrant = QdrantClient(url=settings.qdrant_url)
        self.openai = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.chunker = PolicyChunker()
        self._ensure_collection()

    def _ensure_collection(self):
        if self.qdrant.collection_exists(settings.qdrant_collection):
            return
        self.qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.embedding_dim,
                    distance=models.Distance.COSINE,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True,
                        )
                    ),
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        for field in ("status", "allowed_groups", "policy_id", "department"):
            self.qdrant.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def index_policy(self, raw_text: str, meta: PolicyMetadata, batch_size: int = 64) -> int:
        self.qdrant.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="policy_id", match=models.MatchValue(value=meta.policy_id))]
                )
            ),
        )

        chunks = self.chunker.chunk_document(raw_text, meta)
        if not chunks:
            return 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]

            dense_resp = self.openai.embeddings.create(model=settings.embedding_model, input=texts)
            dense_vecs = [d.embedding for d in dense_resp.data]

            points = []
            for idx, chunk in enumerate(batch):
                sp_indices, sp_values = encode_bm25_sparse(chunk["text"])
                points.append(
                    models.PointStruct(
                        id=chunk["id"],
                        vector={
                            "dense": dense_vecs[idx],
                            "sparse": models.SparseVector(indices=sp_indices, values=sp_values),
                        },
                        payload=chunk["payload"],
                    )
                )
            self.qdrant.upsert(collection_name=settings.qdrant_collection, points=points)

        return len(chunks)