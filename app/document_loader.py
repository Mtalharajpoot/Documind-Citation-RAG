import os
import re
import fitz


def clean_text(text):
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_folder(folder="data/raw_documents"):
    pages = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".pdf"):
            pdf = fitz.open(os.path.join(folder, name))
            for i, page in enumerate(pdf):
                text = clean_text(page.get_text("text", sort=True))
                if len(text) >= 30:
                    pages.append({"document": name, "page": i + 1, "text": text})
            print(name, "->", len(pdf), "pages")
    return pages


if __name__ == "__main__":
    pages = load_folder()
    print("Total pages loaded:", len(pages))
    print(pages[0]["text"][:300])