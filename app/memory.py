import json
import os


class ChatMemory:
    def __init__(self, path="outputs/chat_history.json"):
        self.path = path
        self.items = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.items = json.load(f)
            except Exception:
                self.items = []

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)

    def add(self, question, answer, sources):
        self.items.append({"question": question, "answer": answer, "sources": sources})
        self._save()

    def recent(self, n=3):
        return self.items[-n:]

    def clear(self):
        self.items = []
        self._save()