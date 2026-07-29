from sentence_transformers import SentenceTransformer
import os

print("Loading Embedding Model...")

try:
    # Load only from local cache (No Hugging Face connection)
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True
    )

    print("✓ Embedding Model Loaded Successfully (Local Cache)")

except Exception as e:
    print("\n❌ Failed to load embedding model from local cache.")
    print(e)
    raise RuntimeError(
        "Embedding model not found in local cache. "
        "Download it once when Hugging Face is available."
    )


def embed(chunks):
    """
    Generate embeddings for text chunks.
    """
    return model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )