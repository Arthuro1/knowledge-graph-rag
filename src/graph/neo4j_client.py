"""
Neo4J client — handles connection and Cypher query execution.
"""

import os
from typing import Any
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: dict = {}) -> list[dict[str, Any]]:
        """Run a Cypher query and return results as a list of dicts."""
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]

    def create_constraints(self):
        """Create uniqueness constraints for core node types."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        ]
        for constraint in constraints:
            self.query(constraint)
        print("✅ Neo4J constraints created.")

    def get_graph_stats(self) -> dict:
        """Return basic stats about the current graph."""
        node_count = self.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
        rel_count = self.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
        return {"nodes": node_count, "relationships": rel_count}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
