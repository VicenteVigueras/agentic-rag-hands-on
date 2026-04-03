import asyncio
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# Setup embedding model and vector database
model_embedder = SentenceTransformer("all-mpnet-base-v2")
df = pd.read_csv("semantic_columns.csv")

client = chromadb.Client()
collection = client.create_collection(name="column_semantics")

documents = []
metadatas = []
ids = []

for i, row in df.iterrows():
    text = f"{row['column_name']} column storing {row['column_description']}"
    documents.append(text)
    metadatas.append({
        "column_name": row["column_name"],
        "data_type": row["data_type"]
    })
    ids.append(str(i))

embeddings = model_embedder.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

def search_columns(query, top_k=3):
    """Search for relevant columns based on query"""
    query_embedding = model_embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    context = "Relevant database columns:\n"
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        document = results["documents"][0][i]
        context += f"- {metadata['column_name']} ({metadata['data_type']}): {document}\n"
    
    return context

# LLM setup
model = LiteLlm(
    model="ollama/llama3",
    api_base="http://localhost:11434"
)

async def query_agent(agent, query):
    """Send a query to an agent and return the response"""
    session_service = InMemorySessionService()
    session_id = "session1"
    await session_service.create_session(
        app_name="my_app",
        user_id="user",
        session_id=session_id
    )

    runner = Runner(
        agent=agent,
        app_name="my_app",
        session_service=session_service
    )

    content = Content(role="user", parts=[Part(text=query)])
    final_text = None

    async for event in runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    return final_text


model = LiteLlm(
    model="ollama/llama3",
    api_base="http://localhost:11434"
)


async def main():
    question = "Show me how to find when John made a purchase"
    
    print("=" * 80)
    print("MULTI-HOP REASONING LOOP")
    print("=" * 80)
    print(f"\nInitial Question: {question}\n")
    
    # Create agents for each reasoning step
    discovery_agent = Agent(
        model=model,
        name="discovery_agent",
        description="Identifies what data entities and columns are needed.",
        instruction="Given a question, identify what database entities (tables/columns) would be needed to answer it. List them specifically."
    )
    
    relationship_agent = Agent(
        model=model,
        name="relationship_agent",
        description="Determines how entities connect.",
        instruction="Given entities needed, determine how they relate to each other (joins, foreign keys). Be specific about the relationship."
    )
    
    query_builder_agent = Agent(
        model=model,
        name="query_builder_agent",
        description="Builds the final query.",
        instruction="Given the entities and their relationships, write the SQL query to answer the original question. Output only the SQL."
    )
    
    # LOOP ITERATION 1: Discover needed entities
    print("-" * 80)
    print("LOOP 1: ENTITY DISCOVERY")
    print("-" * 80)
    entities_context = search_columns(question)
    loop1_query = f"{question}\n\nAvailable database columns:\n{entities_context}"
    loop1_output = await query_agent(discovery_agent, loop1_query)
    print(f"Agent: {loop1_output}\n")
    
    # LOOP ITERATION 2: Find relationships (DEPENDS ON LOOP 1)
    print("-" * 80)
    print("LOOP 2: RELATIONSHIP DISCOVERY (uses Loop 1 output)")
    print("-" * 80)
    loop2_query = f"Based on these identified entities, what are the relationships?\n\n{loop1_output}\n\nDatabase context:\n{entities_context}"
    loop2_output = await query_agent(relationship_agent, loop2_query)
    print(f"Agent: {loop2_output}\n")
    
    # LOOP ITERATION 3: Build query (DEPENDS ON LOOPS 1 & 2)
    print("-" * 80)
    print("LOOP 3: SQL QUERY GENERATION (uses Loop 1 & 2 outputs)")
    print("-" * 80)
    loop3_query = f"Original question: {question}\n\nIdentified entities:\n{loop1_output}\n\nRelationships:\n{loop2_output}\n\nNow write the SQL query."
    loop3_output = await query_agent(query_builder_agent, loop3_query)
    print(f"Agent: {loop3_output}\n")
    
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"Original question could be answered with:\n{loop3_output}\n")
    print("Each loop's output was essential for the next loop to function correctly.\n")


asyncio.run(main())
