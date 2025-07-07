# embedder.py

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list) -> list:
        return self.model.encode(texts, show_progress_bar=True)

    def encode_single(self, text: str) -> list:
        return self.model.encode([text])[0]

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
