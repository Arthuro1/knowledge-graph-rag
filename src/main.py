"""
Main CLI entrypoint for Knowledge Graph RAG.

Usage:
    python src/main.py --query "What is machine learning?"
    python src/main.py --query "..." --verbose
"""

import argparse
from src.rag.chain import GraphRAGChain


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph RAG — CLI")
    parser.add_argument("--query", "-q", required=True, help="Your question")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show retrieved context")
    args = parser.parse_args()

    print(f"\n🔍 Query: {args.query}\n")
    print("─" * 60)

    chain = GraphRAGChain()
    try:
        answer = chain.query(args.query, verbose=args.verbose)
        print(f"\n💡 Answer:\n{answer}\n")
    finally:
        chain.close()


if __name__ == "__main__":
    main()
