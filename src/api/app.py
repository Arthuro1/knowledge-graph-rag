"""
FastAPI Application
-------------------
REST API for the Knowledge Graph RAG system.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag.chain import GraphRAGChain
from src.graph.neo4j_client import Neo4jClient

app = FastAPI(
    title="Knowledge Graph RAG API",
    description="RAG system powered by Neo4J Knowledge Graphs and LangChain",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared chain instance
rag_chain = GraphRAGChain()


# -----------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    verbose: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    graph_context: str | None = None


class StatsResponse(BaseModel):
    nodes: int
    relationships: int


# -----------------------------------------------------------------
# Routes
# -----------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Knowledge Graph RAG API is running 🚀"}


@app.get("/stats", response_model=StatsResponse, tags=["Graph"])
def get_graph_stats():
    """Return current graph statistics."""
    try:
        client = Neo4jClient()
        stats = client.get_graph_stats()
        client.close()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entities", tags=["Graph"])
def get_entities(limit: int = Query(default=20, le=100)):
    """Return entities stored in the knowledge graph."""
    try:
        client = Neo4jClient()
        results = client.query(
            """
            MATCH (e:Entity)
            RETURN e.name AS name, e.type AS type, e.description AS description
            ORDER BY e.name
            LIMIT $limit
            """,
            {"limit": limit},
        )
        client.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """Answer a question using the Knowledge Graph RAG pipeline."""
    try:
        context_obj = rag_chain.retriever.retrieve(request.question)
        answer = rag_chain.query(request.question, verbose=request.verbose)
        return QueryResponse(
            question=request.question,
            answer=answer,
            graph_context=context_obj.graph_context if request.verbose else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["RAG"])
def query_stream(request: QueryRequest):
    """Stream the answer token by token."""
    return StreamingResponse(
        rag_chain.stream(request.question),
        media_type="text/plain",
    )
