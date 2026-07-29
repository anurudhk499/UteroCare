import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB = BASE_DIR / "vector_db"

INDEX_PATH = VECTOR_DB / "index.faiss"

CHUNKS_PATH = VECTOR_DB / "chunks.pkl"

# -----------------------------
# Load Embedding Model
# -----------------------------

print("Loading Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load FAISS Index
# -----------------------------

print("Loading FAISS Index...")

index = faiss.read_index(str(INDEX_PATH))

# -----------------------------
# Load Chunks
# -----------------------------

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks.")

# -----------------------------
# Search Function
# -----------------------------

def search(query, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (type exit to quit): ")

        if question.lower() == "exit":
            break

        print("\nRetrieved Knowledge\n")

        docs = search(question)

        for i, doc in enumerate(docs, start=1):

            print("=" * 60)
            print(f"Chunk {i}")
            print("=" * 60)
            print(doc)
            print()