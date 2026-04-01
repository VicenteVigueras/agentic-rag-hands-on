import asyncio
from agent import create_agent, query_agent
from vector_search import search_columns

async def main():
    question = "What columns store information about when a customer made a purchase?"
    
    print("=" * 80)
    print("AGENT VS SENTENCE TRANSFORMERS COMPARISON")
    print("=" * 80)
    print(f"\nQuestion: {question}\n")
    
    print("-" * 80)
    print("1. SENTENCE TRANSFORMERS ALONE (RAG Retrieval)")
    print("-" * 80)
    rag_context = search_columns(question)
    print(rag_context)
    
    print("-" * 80)
    print("2. AGENT WITHOUT RAG CONTEXT")
    print("-" * 80)
    agent_without = create_agent(
        name="agent_without_context",
        instruction="Answer user questions to the best of your knowledge."
    )
    response_without = await query_agent(agent_without, question)
    print(f"Response:\n{response_without}\n")
    
    print("-" * 80)
    print("3. AGENT WITH RAG CONTEXT")
    print("-" * 80)
    agent_with = create_agent(
        name="agent_with_context",
        instruction="Answer user questions to the best of your knowledge. Use the provided database context to give specific, accurate answers."
    )
    enhanced_question = f"{question}\n\n{rag_context}"
    response_with = await query_agent(agent_with, enhanced_question)
    print(f"Response:\n{response_with}\n")
    
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nNotice:")
    print("- Sentence Transformers alone just lists matching columns")
    print("- Agent without context gives generic advice")
    print("- Agent WITH context leverages the retrieved data for intelligent answers\n")

if __name__ == "__main__":
    asyncio.run(main())
