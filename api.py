import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ingestion import PolicyIndexer, PolicyMetadata
from qa_service import PolicyQAService
from retrieval import PolicyRetriever

retriever: PolicyRetriever
qa_service: PolicyQAService
indexer: PolicyIndexer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, qa_service, indexer
    retriever = PolicyRetriever()
    qa_service = PolicyQAService(retriever)
    indexer = PolicyIndexer()
    yield


app = FastAPI(title="Enterprise Policy Q&A Bot", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)


class IngestRequest(BaseModel):
    policy_id: str
    title: str
    department: str
    effective_date: str
    status: str = "active"
    allowed_groups: list[str]
    source_uri: str
    content: str


def get_user_groups(x_user_groups: str = Header(default="all_employees")) -> list[str]:
    """In production, extract verified groups from JWT/OIDC claims."""
    groups = [g.strip() for g in x_user_groups.split(",") if g.strip()]
    if not groups:
        raise HTTPException(status_code=401, detail="Missing user group claims")
    return groups


@app.post("/v1/ask")
async def ask_policy(req: AskRequest, user_groups: list[str] = Depends(get_user_groups)):
    start = time.perf_counter()

    async def event_generator():
        async for line in qa_service.answer_stream(req.query, user_groups):
            yield f"data: {line}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Request-Start": str(start)},
    )


@app.post("/v1/ingest")
async def ingest_policy(req: IngestRequest):
    meta = PolicyMetadata(
        policy_id=req.policy_id,
        title=req.title,
        department=req.department,
        effective_date=req.effective_date,
        status=req.status,
        allowed_groups=req.allowed_groups,
        source_uri=req.source_uri,
    )
    chunk_count = indexer.index_policy(req.content, meta)
    await qa_service.invalidate_cache_on_ingest()
    return {"policy_id": req.policy_id, "chunks_indexed": chunk_count}