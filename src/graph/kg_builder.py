"""
Knowledge Graph Builder
-----------------------
Extracts entities and relationships from raw text using spaCy + LLM,
then writes them into Neo4J.

Usage:
    python src/graph/kg_builder.py --input data/raw/sample_docs.txt
"""

import argparse
import json
import os
from pathlib import Path

import spacy
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.graph.neo4j_client import Neo4jClient

load_dotenv()

# -----------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledge graph extraction expert.
Given a text passage, extract entities and their relationships.

Return ONLY valid JSON with this structure:
{{
  "entities": [
    {{"name": "...", "type": "Concept|Person|Technology|Topic", "description": "..."}}
  ],
  "relationships": [
    {{"source": "...", "relation": "RELATES_TO|REQUIRES|IS_PART_OF|USED_IN", "target": "..."}}
  ]
}}

Be concise. Only extract clear, meaningful relationships."""),
    ("human", "Text: {text}"),
])


# -----------------------------------------------------------------
# Entity & Relation Extraction
# -----------------------------------------------------------------

class KGBuilder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        self.chain = EXTRACTION_PROMPT | self.llm
        self.client = Neo4jClient()

    def extract(self, text: str) -> dict:
        """Extract entities and relations from a text chunk via LLM."""
        response = self.chain.invoke({"text": text})
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    def chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """Split text into overlapping chunks by sentence."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        chunks, current, count = [], [], 0
        for sent in sentences:
            current.append(sent)
            count += len(sent.split())
            if count >= chunk_size:
                chunks.append(" ".join(current))
                current = current[-2:]  # overlap: keep last 2 sentences
                count = sum(len(s.split()) for s in current)

        if current:
            chunks.append(" ".join(current))
        return chunks

    def write_to_neo4j(self, extracted: dict, source: str = "unknown"):
        """Write extracted entities and relationships into Neo4J."""
        # Write entities
        for entity in extracted.get("entities", []):
            self.client.query(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.description = $description,
                    e.source = $source
                """,
                {
                    "name": entity["name"],
                    "type": entity.get("type", "Concept"),
                    "description": entity.get("description", ""),
                    "source": source,
                },
            )

        # Write relationships
        for rel in extracted.get("relationships", []):
            self.client.query(
                f"""
                MATCH (a:Entity {{name: $source}})
                MATCH (b:Entity {{name: $target}})
                MERGE (a)-[r:{rel['relation']}]->(b)
                """,
                {"source": rel["source"], "target": rel["target"]},
            )

    def build_from_file(self, filepath: str):
        """Full pipeline: read file → chunk → extract → write to Neo4J."""
        text = Path(filepath).read_text(encoding="utf-8")
        chunks = self.chunk_text(text)
        filename = Path(filepath).name

        print(f"📄 Processing '{filename}' — {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            print(f"  → Chunk {i + 1}/{len(chunks)} ...", end=" ")
            try:
                extracted = self.extract(chunk)
                self.write_to_neo4j(extracted, source=filename)
                entities = len(extracted.get("entities", []))
                rels = len(extracted.get("relationships", []))
                print(f"✓ {entities} entities, {rels} relations")
            except Exception as e:
                print(f"✗ Error: {e}")

        stats = self.client.get_graph_stats()
        print(f"\n✅ Graph updated: {stats['nodes']} nodes, {stats['relationships']} relationships")

    def close(self):
        self.client.close()


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from text files")
    parser.add_argument("--input", required=True, help="Path to input text file")
    args = parser.parse_args()

    builder = KGBuilder()
    try:
        builder.build_from_file(args.input)
    finally:
        builder.close()
