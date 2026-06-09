"""
Hybrid Retriever
----------------
Combines Neo4J graph traversal with semantic vector search
to produce richer context for the LLM.
"""

import os
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

from src.graph.neo4j_client import Neo4jClient

load_dotenv()


@dataclass
class RetrievedContext:
    graph_context: str
    vector_context: str
    combined: str


# -----------------------------------------------------------------
# Vector Store
# -----------------------------------------------------------------

class VectorStore:
    def __init__(self, collection_name: str = "kg_rag_docs"):
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self.client = chromadb.PersistentClient(path="data/processed/chroma")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

    def add_texts(self, texts: list[str], ids: list[str] | None = None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.upsert(documents=texts, ids=ids)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results["documents"][0] if results["documents"] else []


# -----------------------------------------------------------------
# Graph Retriever
# -----------------------------------------------------------------

class GraphRetriever:
    """Retrieves relevant subgraph context using Cypher queries."""

    def __init__(self):
        self.client = Neo4jClient()

    def get_related_entities(self, entity_name: str, depth: int = 2) -> list[dict]:
        """Find entities connected to a given entity up to N hops away."""
        cypher = """
        MATCH path = (start:Entity {name: $name})-[*1..$depth]-(related:Entity)
        RETURN related.name AS name,
               related.type AS type,
               related.description AS description,
               length(path) AS distance
        ORDER BY distance
        LIMIT 20
        """
        return self.client.query(cypher, {"name": entity_name, "depth": depth})

    def get_context_for_query(self, query: str) -> str:
        """
        Simple keyword-based graph lookup.
        In production, replace with NER to extract entities from the query.
        """
        words = [w for w in query.split() if len(w) > 4]
        results = []

        for word in words[:3]:  # Limit to top 3 keywords
            cypher = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($keyword)
            OPTIONAL MATCH (e)-[r]->(related:Entity)
            RETURN e.name AS entity,
                   e.description AS description,
                   type(r) AS relation,
                   related.name AS related_entity
            LIMIT 10
            """
            rows = self.client.query(cypher, {"keyword": word})
            results.extend(rows)

        if not results:
            return "No relevant graph context found."

        lines = []
        for row in results:
            line = f"- {row['entity']}: {row.get('description', '')}"
            if row.get("related_entity"):
                line += f" → [{row['relation']}] → {row['related_entity']}"
            lines.append(line)

        return "\n".join(lines)

    def close(self):
        self.client.close()


# -----------------------------------------------------------------
# Hybrid Retriever
# -----------------------------------------------------------------

class HybridRetriever:
    """
    Combines graph traversal context with vector similarity search
    to give the LLM the best of both worlds.
    """

    def __init__(self):
        self.graph_retriever = GraphRetriever()
        self.vector_store = VectorStore()
        self.top_k = int(os.getenv("TOP_K_RESULTS", 5))

    def retrieve(self, query: str) -> RetrievedContext:
        graph_ctx = self.graph_retriever.get_context_for_query(query)
        vector_docs = self.vector_store.search(query, top_k=self.top_k)
        vector_ctx = "\n\n".join(vector_docs) if vector_docs else "No vector results found."

        combined = f"""### Knowledge Graph Context:
{graph_ctx}

### Semantic Search Context:
{vector_ctx}"""

        return RetrievedContext(
            graph_context=graph_ctx,
            vector_context=vector_ctx,
            combined=combined,
        )

    def close(self):
        self.graph_retriever.close()
