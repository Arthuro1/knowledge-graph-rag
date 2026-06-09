"""
Tests for the KG Builder.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch


class TestKGBuilder:
    @patch("src.graph.kg_builder.Neo4jClient")
    @patch("src.graph.kg_builder.ChatOpenAI")
    def test_chunk_text(self, mock_llm, mock_client):
        from src.graph.kg_builder import KGBuilder
        builder = KGBuilder.__new__(KGBuilder)
        import spacy
        builder.nlp = spacy.load("en_core_web_sm")

        text = "Machine learning is a subset of AI. " * 30
        chunks = builder.chunk_text(text, chunk_size=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0

    @patch("src.graph.kg_builder.Neo4jClient")
    @patch("src.graph.kg_builder.ChatOpenAI")
    def test_extract_returns_dict(self, mock_llm, mock_client):
        from src.graph.kg_builder import KGBuilder
        builder = KGBuilder.__new__(KGBuilder)

        mock_response = MagicMock()
        mock_response.content = '{"entities": [{"name": "Python", "type": "Technology", "description": "A programming language"}], "relationships": []}'
        builder.chain = MagicMock(return_value=mock_response)
        builder.chain.invoke = MagicMock(return_value=mock_response)

        result = builder.extract("Python is a programming language.")
        assert "entities" in result
        assert "relationships" in result


class TestHybridRetriever:
    @patch("src.rag.retriever.Neo4jClient")
    @patch("src.rag.retriever.chromadb.PersistentClient")
    def test_retrieve_returns_context(self, mock_chroma, mock_neo4j):
        from src.rag.retriever import HybridRetriever, RetrievedContext

        mock_neo4j.return_value.query.return_value = []
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [["Some context"]]}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        retriever = HybridRetriever()
        result = retriever.retrieve("What is machine learning?")

        assert isinstance(result, RetrievedContext)
        assert isinstance(result.combined, str)
