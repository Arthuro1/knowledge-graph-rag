"""
Graph RAG Chain
---------------
Combines the HybridRetriever with an LLM to answer questions
grounded in the Knowledge Graph + vector context.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.rag.retriever import HybridRetriever

load_dotenv()

# -----------------------------------------------------------------
# LLM Provider selection
# -----------------------------------------------------------------

def get_llm():
    """Auto-select LLM based on available API keys in .env."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
        print(f"🤖 Using Anthropic: {model}")
        return ChatAnthropic(model=model, temperature=0, anthropic_api_key=anthropic_key)

    if openai_key and openai_key != "your_openai_api_key_here":
        from langchain_openai import ChatOpenAI
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        print(f"🤖 Using OpenAI: {model}")
        return ChatOpenAI(model=model, temperature=0)

    raise ValueError(
        "❌ No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
    )

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
        self.llm = get_llm()
        self.chain = RAG_PROMPT | self.llm | StrOutputParser()

    def query(self, question: str, verbose: bool = True) -> str:
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
