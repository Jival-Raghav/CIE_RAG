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


## Embedding Models needed for the reranker.py
## Note for Team: These are some models which might be needed for the multimodal team, please do make necessary changes- if they are using a different model
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class EmbeddingModelLoader:
    # Team, Make necessary Changes
    def __init__(self):
        self.image_processor = None
        self.image_model = None
        self.text_tokenizer = None
        self.text_model = None

    def load_image_model(self, model_name="Salesforce/blip-image-captioning-base"):
        self.image_processor = BlipProcessor.from_pretrained(model_name)
        self.image_model = BlipForConditionalGeneration.from_pretrained(model_name).eval()

    def load_text_model(self, model_name="BAAI/bge-reranker-base"):
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def load_all(self):
        self.load_image_model()
        self.load_text_model()

    def get_image_components(self):
        return self.image_processor, self.image_model

    def get_text_components(self):
        return self.text_tokenizer, self.text_model
