import pickle
import faiss
import numpy as np

from pathlib import Path

from .pdf_loader import load_pdf
from .chunker import chunk_text
from .embeddings import embed

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "knowledge_base" / "uterocare_medical_knowledge.pdf"

VECTOR_DB = BASE_DIR / "vector_db"

VECTOR_DB.mkdir(exist_ok=True)

print("Loading PDF...")

text = load_pdf(PDF_PATH)

print("Chunking...")

chunks = chunk_text(text)

print("Creating Embeddings...")

vectors = embed(chunks)

print("Creating FAISS Index...")

dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(vectors))

print("Saving...")

faiss.write_index(
    index,
    str(VECTOR_DB / "index.faiss")
)

with open(
    VECTOR_DB / "chunks.pkl",
    "wb"
) as f:
    pickle.dump(chunks, f)

print(f"Knowledge Base Ready ({len(chunks)} chunks)")