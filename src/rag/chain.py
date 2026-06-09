"""
Graph RAG Chain
---------------
Combines the HybridRetriever with an LLM to answer questions
grounded in the Knowledge Graph + vector context.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.rag.retriever import HybridRetriever

load_dotenv()

# -----------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to a Knowledge Graph.
Use the provided context to answer the user's question accurately.

Guidelines:
- Prioritize Knowledge Graph context for factual relationships
- Use Semantic Search context for detailed explanations
- If the context doesn't contain the answer, say so clearly
- Be concise but complete

Context:
{context}
"""),
    ("human", "{question}"),
])


# -----------------------------------------------------------------
# Chain
# -----------------------------------------------------------------

class GraphRAGChain:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )
        self.chain = RAG_PROMPT | self.llm | StrOutputParser()

    def query(self, question: str, verbose: bool = False) -> str:
        """
        Run a full RAG query:
        1. Retrieve hybrid context (graph + vector)
        2. Generate answer with LLM
        """
        context_obj = self.retriever.retrieve(question)

        if verbose:
            print("\n📊 Graph Context:")
            print(context_obj.graph_context)
            print("\n📄 Vector Context:")
            print(context_obj.vector_context)
            print()

        answer = self.chain.invoke({
            "context": context_obj.combined,
            "question": question,
        })
        return answer

    def stream(self, question: str):
        """Stream the answer token by token (useful for APIs/UIs)."""
        context_obj = self.retriever.retrieve(question)
        stream_chain = RAG_PROMPT | self.llm | StrOutputParser()
        yield from stream_chain.stream({
            "context": context_obj.combined,
            "question": question,
        })

    def close(self):
        self.retriever.close()
