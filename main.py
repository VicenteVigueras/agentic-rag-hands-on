import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load dataset
# -------------------------------

df = pd.read_csv("semantic_columns.csv")

# -------------------------------
# Load embedding model
# -------------------------------

# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("all-mpnet-base-v2")


# -------------------------------
# Initialize Chroma
# -------------------------------

client = chromadb.Client()

collection = client.create_collection(
    name="column_semantics"
)

# -------------------------------
# Insert rows into vector database
# -------------------------------

documents = []
metadatas = []
ids = []

for i, row in df.iterrows():

    # Combine column name + description for better embeddings
    text = f"{row['column_name']} column storing {row['column_description']}"

    documents.append(text)

    metadatas.append({
        "column_name": row["column_name"],
        "data_type": row["data_type"]
    })

    ids.append(str(i))

# Generate embeddings
embeddings = model.encode(documents).tolist()

# Insert into Chroma
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("Dataset successfully indexed.")

# -------------------------------
# Semantic search function
# -------------------------------

def search_columns(query, top_k=5):

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    print("\nQuery:", query)
    print("\nTop Results:\n")

    for i in range(len(results["ids"][0])):

        metadata = results["metadatas"][0][i]
        document = results["documents"][0][i]
        distance = results["distances"][0][i]

        print("Column:", metadata["column_name"])
        print("Type:", metadata["data_type"])
        print("Description:", document)
        print("Distance:", round(distance, 4))
        print("-" * 40)


# -------------------------------
# Example searches
# -------------------------------

search_columns("when did the customer purchase their item")
search_columns("how much did the customer pay")
