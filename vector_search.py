import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
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

embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

def search_columns(query, top_k=3):
    """Search for relevant columns based on query"""
    query_embedding = model.encode([query]).tolist()
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
