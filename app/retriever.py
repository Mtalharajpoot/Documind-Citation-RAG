import json
import faiss
from sentence_transformers import SentenceTransformer
from vector_store import MODEL_NAME, INDEX_DIR


class Retriever:
    def __init__(self, name="config1", top_k=3):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(f"{INDEX_DIR}/{name}.faiss")
        with open(f"{INDEX_DIR}/{name}_chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.top_k = top_k

    def search(self, question, top_k=None):
        k = top_k or self.top_k
        q = self.model.encode([question], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        results = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            c = dict(self.chunks[i])
            c["score"] = round(float(score), 3)
            results.append(c)
        return results


if __name__ == "__main__":
    r = Retriever("config1", top_k=3)
    questions = [
        "How many attention heads does the base Transformer use?",
        "What is the masked language model objective in BERT?",
        "How does RAG combine a retriever and a generator?",
        "Who is the CEO of Google?",
    ]
    for q in questions:
        print("\nQUESTION:", q)
        for c in r.search(q):
            print(f"  [{c['score']}] {c['document']} | page {c['page']}")
            print("     ", c["text"][:150], "...")