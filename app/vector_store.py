import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from document_loader import load_folder
from text_chunker import chunk_pages

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "vector_db/index_files"


def build_index(model, pages, chunk_size, overlap, name):
    os.makedirs(INDEX_DIR, exist_ok=True)
    chunks = chunk_pages(pages, chunk_size, overlap)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.array(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # normalized vectors par inner product = cosine similarity
    index.add(emb)

    faiss.write_index(index, f"{INDEX_DIR}/{name}.faiss")
    with open(f"{INDEX_DIR}/{name}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"{name}: {len(chunks)} chunks, vector size {emb.shape[1]}, saved.")


if __name__ == "__main__":
    pages = load_folder()
    model = SentenceTransformer(MODEL_NAME)
    build_index(model, pages, 500, 100, "config1")
    build_index(model, pages, 1000, 150, "config2")