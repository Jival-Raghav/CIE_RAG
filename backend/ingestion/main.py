from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
import os
import uvicorn
import time
from typing import List, Optional

from qdrant_database import QdrantManager
from faiss_database import setup_faiss_with_text_storage
from llama_index.core.schema import TextNode, Document as LLDocument
from BBCB_modules import Parser

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
parser = Parser()

# Initialize database managers
qdrant_manager = QdrantManager(collection_name="docs")
faiss_retriever = None

@app.get("/")
def health():
    return {"status": "Welcome to the vector database, FastAPI is running without any issues."}

def process_file(file_path: str, question: str = "Summarize this file") -> dict:
    """Process a single file using the parser and return extracted content"""
    try:
        text, images, transcript, answer, video_clip, matched_content = parser.process_document(
            uploaded_file=file_path,
            gdrive_url=None,
            question=question
        )
        
        result = {
            "text": text,
            "images": images,
            "transcript": transcript,
            "answer": answer,
            "video_clip": video_clip,
            "matched_content": matched_content
        }
        
        # Save video clip if exists
        if video_clip and os.path.exists(video_clip):
            result["video_clip_url"] = f"/download/{os.path.basename(video_clip)}"
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def parse_and_index_dir(upload_dir: str) -> List[LLDocument]:
    """Process all supported files in a directory and create documents for indexing"""
    documents = []
    for root, dirs, files in os.walk(upload_dir):
        for fname in files:
            file_path = os.path.join(root, fname)
            file_ext = os.path.splitext(fname)[-1].lower()
            
            if file_ext in [".pdf", ".pptx", ".docx", ".xlsx"]:
                try:
                    result = process_file(file_path)
                    if result["text"] and isinstance(result["text"], str) and result["text"].strip():
                        metadata = {
                            "file": fname,
                            "file_path": file_path,
                            "images": result["images"],
                            "transcript": result["transcript"]
                        }
                        doc = LLDocument(
                            text=result["text"].strip(),
                            metadata=metadata
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"Error processing file {fname}: {str(e)}")
                    continue
    return documents

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    question: str = Form("Summarize this file"),
    index_to_qdrant: bool = Form(True),
    index_to_faiss: bool = Form(True)
):
    try:
        # Validate file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in [".pdf", ".pptx", ".mp4", ".avi", ".mov", ".mkv", ".docx", ".xlsx"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        # Save uploaded file
        upload_dir = "uploaded_files"
        os.makedirs(upload_dir, exist_ok=True)
        target_path = os.path.join(upload_dir, file.filename)

        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Process the file
        process_result = process_file(target_path, question)
        
        # Indexing results
        indexing_results = {}
        
        # Index to Qdrant if enabled
        if index_to_qdrant and qdrant_manager.client:
            try:
                doc = LLDocument(
                    text=process_result["text"],
                    metadata={
                        "file": file.filename,
                        "file_path": target_path,
                        "images": process_result["images"],
                        "transcript": process_result["transcript"],
                        "file_type": ext[1:].upper()
                    }
                )
                
                chunks = [{
                    "text": doc.text,
                    "source": file.filename,
                    "file_type": ext[1:].upper()
                }]
                
                qdrant_success = qdrant_manager.store_documents(chunks)
                indexing_results["qdrant"] = "success" if qdrant_success else "failed"
            except Exception as e:
                indexing_results["qdrant"] = f"failed: {str(e)}"
        
        # Index to FAISS if enabled
        if index_to_faiss:
            try:
                global faiss_retriever
                documents = parse_and_index_dir(upload_dir)
                if documents:
                    nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]
                    faiss_retriever, faiss_time = setup_faiss_with_text_storage(nodes)
                    indexing_results["faiss"] = f"success ({faiss_time:.2f}s)"
                else:
                    indexing_results["faiss"] = "no documents to index"
            except Exception as e:
                indexing_results["faiss"] = f"failed: {str(e)}"

        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "processing_result": {
                "answer": process_result["answer"],
                "images_count": len(process_result["images"]),
                "has_transcript": bool(process_result["transcript"]),
                "has_video_clip": bool(process_result.get("video_clip_url", False))
            },
            "indexing_results": indexing_results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Endpoint to download processed files (video clips, etc.)"""
    file_path = None
    
    # Check in different possible locations
    possible_locations = [
        f"components/clips/{filename}",
        f"components/videos/{filename}",
        f"components/images/{filename}",
        f"uploaded_files/{filename}"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            file_path = location
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.post("/search/")
async def search_doc(
    query: str = Form(...),
    top_k: int = Form(5),
    use_qdrant: bool = Form(True),
    use_faiss: bool = Form(True)
):
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        results = []
        sources_used = []
        
        # Try Qdrant first if enabled
        if use_qdrant and qdrant_manager.client:
            try:
                q_results = qdrant_manager.search(query=query, limit=top_k)
                if q_results:
                    results.extend([{
                        "text": r["text"],
                        "metadata": r["metadata"],
                        "score": r["score"],
                        "source": "qdrant"
                    } for r in q_results])
                    sources_used.append("qdrant")
            except Exception as e:
                print(f"Qdrant search error: {str(e)}")

        # Fallback to FAISS if enabled and no Qdrant results
        if use_faiss and (not results or len(results) < top_k) and faiss_retriever:
            try:
                # Get additional results needed
                additional_k = max(top_k - len(results), 1)
                f_results = faiss_retriever.retrieve(query, top_k=additional_k)
                
                if f_results:
                    results.extend([{
                        "text": r.text,
                        "metadata": r.metadata,
                        "score": r.score,
                        "source": "faiss"
                    } for r in f_results])
                    sources_used.append("faiss")
            except Exception as e:
                print(f"FAISS search error: {str(e)}")

        if not results:
            raise HTTPException(status_code=404, detail="No results found")

        # Sort all results by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "results": results[:top_k],
            "sources_used": sources_used
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

