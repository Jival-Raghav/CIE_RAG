# retriever.py

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search_documents(self, query: str, top_k: int = 5):
        return self.vector_store.search(query, limit=top_k)

    def format_results(self, results):
        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r['score']:.3f} | Source: {r['source']}")
            print(f"   Text: {r['text'][:200]}...")
            print("-" * 50)
