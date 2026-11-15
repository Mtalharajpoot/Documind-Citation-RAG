from document_loader import load_folder


def chunk_pages(pages, chunk_size=500, overlap=100):
    chunks = []
    for p in pages:
        text = p["text"]
        doc_stem = p["document"].replace(".pdf", "")
        start, n = 0, 1
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start + chunk_size // 2:
                    end = space
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "chunk_id": f"{doc_stem}_page{p['page']}_chunk{n}",
                    "document": p["document"],
                    "page": p["page"],
                    "text": piece,
                })
                n += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


if __name__ == "__main__":
    pages = load_folder()
    for size, ov in [(500, 100), (1000, 150)]:
        chunks = chunk_pages(pages, size, ov)
        print(f"chunk_size={size}, overlap={ov} -> {len(chunks)} chunks")
    chunks = chunk_pages(pages, 500, 100)
    print()
    print(chunks[3])