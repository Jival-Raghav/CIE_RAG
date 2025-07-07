'''
Ok so this is the backend that handles the vector databases by keeping faiss and qdrant in sync, where faiss is to fallback upon if 
qdrant is not available.

To run this code create a .env file with the following variables:
MISTRAL_API_KEY=your_mistral_api_key
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=documents
'''

from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import shutil
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from data_loader import *
from embedder import *
from load_docs import *
from retriever import *
from parser2 import *
from faissdbb import *

# Load environment variables (for API keys etc.)
load_dotenv()

app = FastAPI()

# Initialize systems
qdrant = QdrantManager(
    collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY")
)

faissdb = FAISSVectorDB(db_path="./my_vector_db")

parser = Parser(
    mistral_api_key=os.getenv('MISTRAL_API_KEY'),
    qdrant_url=os.getenv("QDRANT_URL")
)

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.csv', '.json', '.txt', '.pptx', '.mp4', '.avi', '.mov', '.mkv'}

@app.post("/")
def health():
    return {"status": "🟢 fastapi is running"}

@app.post("/create_DB/")
def creating():
    faiss_status = "✅"
    qdrant_status = "✅"

    try:
        faissdb.setup_index(index_type="flat", force_recreate=True)
    except Exception as e:
        faiss_status = f"❌ FAISS error: {str(e)}"

    try:
        qdrant._setup_collection()
    except Exception as e:
        qdrant_status = f"❌ Qdrant error: {str(e)}"

    return {
        "faiss_status": faiss_status,
        "qdrant_status": qdrant_status
    }

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    tmp_path = None
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return {"error": f"Unsupported file type: {ext}"}

        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Use parser2 for advanced parsing (especially for images, videos, captions)
        parser_output = []
        if ext in [".pdf", ".pptx", ".mp4", ".avi", ".mov", ".mkv"]:
            parser_output = parser.extract_and_index_files(str(Path(tmp_path).parent))
            return {
                "message": "Processed via parser2 pipeline",
                "file": file.filename
            }

        # Use QdrantManager for plain text/doc/csv/json
        chunks = qdrant.process_file(tmp_path)
        if not chunks:
            return {"message": "No data extracted."}

        stored_qdrant = qdrant.store_documents(chunks)

        stored_faiss = False
        try:
            texts = [chunk['text'] for chunk in chunks]
            metadata = [chunk for chunk in chunks]
            ids = [f"{chunk['file_type']}_{chunk['chunk_id']}" for chunk in chunks]
            faissdb.add_texts(texts, metadata=metadata, ids=ids)
            stored_faiss = True
        except Exception as e:
            stored_faiss = f"❌ Error storing in FAISS: {str(e)}"

        return {
            "qdrant_stored": stored_qdrant,
            "faiss_stored": stored_faiss,
            "chunks_count": len(chunks)
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/retrieve/")
async def retrieve(query: str = Form(...)):
    try:
        if not query.strip():
            return {"error": "Query cannot be empty"}

        # Qdrant First
        qdrant_results = qdrant.search(query)
        if qdrant_results:
            return {"source": "qdrant", "results": qdrant_results}

        # FAISS fallback
        faiss_results = faissdb.search(query, k=5)
        return {"source": "faiss", "results": faiss_results if faiss_results else "No results found in either DB"}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
