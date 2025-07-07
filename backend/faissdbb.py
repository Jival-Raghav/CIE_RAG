#!/usr/bin/env python3
"""
FAISS Vector Database System
A complete system for embedding storage and nearest-neighbor search using FAISS.

IMPORTANT: Do NOT name this file 'faiss.py' as it will cause import conflicts!
Save as 'vector_db.py' or 'faiss_database.py' instead.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

try:
    import faiss
except ImportError:
    print("Please install FAISS: pip install faiss-cpu")
    print("For GPU support: pip install faiss-gpu")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSVectorDB:
    """
    A comprehensive FAISS-based vector database for embedding storage and similarity search.
    """
    
    def __init__(self, db_path: str = "./vector_db", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the FAISS Vector Database.
        
        Args:
            db_path: Path to store the database files
            embedding_model: Name of the sentence transformer model to use
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.id_mapping = {}  # Map external IDs to internal indices
        self.next_id = 0
        
        # Files for persistence
        self.index_file = self.db_path / "faiss.index"
        self.metadata_file = self.db_path / "metadata.json"
        self.mapping_file = self.db_path / "id_mapping.pkl"
        
        # Load existing database if available
        self.load_database()
    
    def _create_index(self, index_type: str = "flat") -> faiss.Index:
        """
        Create a FAISS index based on the specified type.
        
        Args:
            index_type: Type of index ('flat', 'ivf', 'hnsw')
        """
        if index_type == "flat":
            # Exact search using L2 distance
            return faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "ivf":
            # Inverted file index for faster approximate search
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            nlist = 100  # Number of clusters
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World for very fast approximate search
            M = 16  # Number of connections
            return faiss.IndexHNSWFlat(self.embedding_dim, M)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def setup_index(self, index_type: str = "flat", force_recreate: bool = False):
        """
        Setup or recreate the FAISS index.
        
        Args:
            index_type: Type of index to create
            force_recreate: Whether to recreate even if index exists
        """
        if self.index is None or force_recreate:
            logger.info(f"Creating new {index_type} index")
            self.index = self._create_index(index_type)
            
            # For IVF index, we need to train it
            if index_type == "ivf" and len(self.metadata) > 0:
                # Re-encode all existing texts for training
                texts = [item['text'] for item in self.metadata]
                embeddings = self.encoder.encode(texts)
                self.index.train(embeddings.astype('float32'))
                self.index.add(embeddings.astype('float32'))
    
    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None, 
                  ids: Optional[List[str]] = None) -> List[int]:
        """
        Add texts to the vector database.
        
        Args:
            texts: List of texts to add
            metadata: Optional metadata for each text
            ids: Optional external IDs for each text
        
        Returns:
            List of internal IDs assigned to the texts
        """
        if self.index is None:
            self.setup_index()
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(texts)
        
        # Assign IDs
        if ids is None:
            ids = [f"doc_{self.next_id + i}" for i in range(len(texts))]
        
        internal_ids = []
        for i, (text, meta, ext_id) in enumerate(zip(texts, metadata, ids)):
            internal_id = len(self.metadata)
            internal_ids.append(internal_id)
            
            # Store metadata
            meta_entry = {
                'text': text,
                'external_id': ext_id,
                'internal_id': internal_id,
                'timestamp': datetime.now().isoformat(),
                **meta
            }
            self.metadata.append(meta_entry)
            self.id_mapping[ext_id] = internal_id
            
            self.next_id = max(self.next_id, internal_id + 1)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Train IVF index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training IVF index")
            self.index.train(embeddings.astype('float32'))
            self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Added {len(texts)} texts to database")
        return internal_ids
    
    def search(self, query: str, k: int = 5, return_metadata: bool = True) -> List[Dict]:
        """
        Search for similar texts in the database.
        
        Args:
            query: Query text
            k: Number of results to return
            return_metadata: Whether to include metadata in results
        
        Returns:
            List of search results with scores and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.encoder.encode([query])
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
                
            result = {
                'score': float(score),
                'internal_id': int(idx),
                'text': self.metadata[idx]['text']
            }
            
            if return_metadata:
                result['metadata'] = self.metadata[idx]
            
            results.append(result)
        
        return results
    
    def search_by_vector(self, vector: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search using a pre-computed vector.
        
        Args:
            vector: Query vector
            k: Number of results to return
        
        Returns:
            List of search results
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        scores, indices = self.index.search(vector.reshape(1, -1).astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            results.append({
                'score': float(score),
                'internal_id': int(idx),
                'text': self.metadata[idx]['text'],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def get_by_id(self, external_id: str) -> Optional[Dict]:
        """Get document by external ID."""
        if external_id not in self.id_mapping:
            return None
        
        internal_id = self.id_mapping[external_id]
        return self.metadata[internal_id]
    
    def delete_by_id(self, external_id: str) -> bool:
        """
        Delete document by external ID.
        Note: FAISS doesn't support deletion, so this marks as deleted.
        """
        if external_id not in self.id_mapping:
            return False
        
        internal_id = self.id_mapping[external_id]
        self.metadata[internal_id]['deleted'] = True
        return True
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        active_docs = len([m for m in self.metadata if not m.get('deleted', False)])
        
        return {
            'total_documents': len(self.metadata),
            'active_documents': active_docs,
            'deleted_documents': len(self.metadata) - active_docs,
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.index).__name__ if self.index else None,
            'index_size': self.index.ntotal if self.index else 0
        }
    
    def save_database(self):
        """Persist the database to disk."""
        if self.index is not None:
            logger.info("Saving FAISS index")
            faiss.write_index(self.index, str(self.index_file))
        
        logger.info("Saving metadata")
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        with open(self.mapping_file, 'wb') as f:
            pickle.dump({
                'id_mapping': self.id_mapping,
                'next_id': self.next_id
            }, f)
    
    def load_database(self):
        """Load existing database from disk."""
        # Load FAISS index
        if self.index_file.exists():
            logger.info("Loading FAISS index")
            self.index = faiss.read_index(str(self.index_file))
        
        # Load metadata
        if self.metadata_file.exists():
            logger.info("Loading metadata")
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Load ID mapping
        if self.mapping_file.exists():
            with open(self.mapping_file, 'rb') as f:
                data = pickle.load(f)
                self.id_mapping = data['id_mapping']
                self.next_id = data['next_id']
    
    def clear_database(self):
        """Clear all data from the database."""
        self.index = None
        self.metadata = []
        self.id_mapping = {}
        self.next_id = 0
        
        # Remove files
        for file_path in [self.index_file, self.metadata_file, self.mapping_file]:
            if file_path.exists():
                file_path.unlink()

def load_documents_from_folder(folder_path: str) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Load documents from a folder containing text files.
    
    Args:
        folder_path: Path to the folder containing documents
    
    Returns:
        Tuple of (texts, metadata, ids)
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    documents = []
    metadata = []
    ids = []
    
    # Supported file extensions
    supported_extensions = {'.txt', '.md', '.doc', '.docx', '.pdf'}
    
    print(f"Loading documents from: {folder_path}")
    
    # Process different file types
    for file_path in folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                content = ""
                file_id = file_path.stem  # filename without extension
                
                if file_path.suffix.lower() == '.txt' or file_path.suffix.lower() == '.md':
                    # Plain text files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                
                elif file_path.suffix.lower() == '.pdf':
                    # PDF files (requires PyPDF2 or similar)
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            content = ""
                            for page in pdf_reader.pages:
                                content += page.extract_text() + "\n"
                    except ImportError:
                        print(f"Warning: PyPDF2 not installed. Skipping PDF: {file_path.name}")
                        continue
                    except Exception as e:
                        print(f"Error reading PDF {file_path.name}: {e}")
                        continue
                
                elif file_path.suffix.lower() in ['.doc', '.docx']:
                    # Word documents (requires python-docx)
                    try:
                        from docx import Document
                        doc = Document(file_path)
                        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    except ImportError:
                        print(f"Warning: python-docx not installed. Skipping Word doc: {file_path.name}")
                        continue
                    except Exception as e:
                        print(f"Error reading Word document {file_path.name}: {e}")
                        continue
                
                if content and len(content.strip()) > 0:
                    documents.append(content)
                    metadata.append({
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'file_extension': file_path.suffix,
                        'file_size': file_path.stat().st_size,
                        'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
                    ids.append(file_id)
                    print(f"  ✓ Loaded: {file_path.name}")
                else:
                    print(f"  ⚠ Empty file skipped: {file_path.name}")
                    
            except Exception as e:
                print(f"  ✗ Error loading {file_path.name}: {e}")
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents, metadata, ids

def main():
    """Main function to load documents from folder and setup vector database."""
    
    # Initialize database
    print("Initializing FAISS Vector Database...")
    db = FAISSVectorDB(db_path="./my_vector_db")
    
    # Specify your documents folder path
    documents_folder = input("Enter the path to your documents folder (or press Enter for './documents'): ").strip()
    if not documents_folder:
        documents_folder = "./documents"
    
    try:
        # Load documents from folder
        documents, metadata, ids = load_documents_from_folder(documents_folder)
        
        if not documents:
            print("No documents found in the specified folder!")
            print("Please ensure your folder contains .txt, .md, .pdf, .doc, or .docx files")
            return
        
        # Clear existing database if user wants to
        if db.get_stats()['total_documents'] > 0:
            clear_db = input("Existing database found. Clear it and add new documents? (y/N): ").strip().lower()
            if clear_db == 'y':
                db.clear_database()
                print("Previous database cleared.")
        
        # Add documents to database
        print(f"\nAdding {len(documents)} documents to database...")
        db.add_texts(documents, metadata=metadata, ids=ids)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create a 'documents' folder and add your text files, or specify a valid path.")
        return
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    
    
    # Save database
    db.save_database()
    
    # Print statistics
    stats = db.get_stats()
    print(f"\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Interactive search loop
    print(f"\n" + "="*50)
    print("INTERACTIVE SEARCH MODE")
    print("="*50)
    print("Enter your search queries (type 'quit' to exit)")
    
    while True:
        query = input(f"\nSearch query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
            
        print(f"\nSearching for: '{query}'")
        results = db.search(query, k=5)
        
        if not results:
            print("No results found.")
            continue
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   File: {result['metadata']['filename']}")
            print(f"   Text preview: {result['text'][:200]}...")
            if len(result['text']) > 200:
                print(f"   [Text truncated - full length: {len(result['text'])} characters]")
    
    print(f"\nDatabase saved. You can run the program again to search your documents!")

# Additional utility functions
def add_documents_to_existing_db(db_path: str = "./my_vector_db", documents_folder: str = "./documents"):
    """Add documents to an existing database without clearing it."""
    db = FAISSVectorDB(db_path=db_path)
    
    try:
        documents, metadata, ids = load_documents_from_folder(documents_folder)
        if documents:
            db.add_texts(documents, metadata=metadata, ids=ids)
            db.save_database()
            print(f"Added {len(documents)} new documents to existing database.")
        else:
            print("No documents found to add.")
    except Exception as e:
        print(f"Error: {e}")

def search_database(db_path: str = "./my_vector_db"):
    """Search an existing database interactively."""
    db = FAISSVectorDB(db_path=db_path)
    
    stats = db.get_stats()
    if stats['total_documents'] == 0:
        print("No documents in database. Please add documents first.")
        return
    
    print(f"Database loaded with {stats['active_documents']} documents.")
    print("Enter your search queries (type 'quit' to exit)")
    
    while True:
        query = input(f"\nSearch query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
            
        results = db.search(query, k=5)
        
        if not results:
            print("No results found.")
            continue
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   File: {result['metadata']['filename']}")
            print(f"   Text preview: {result['text'][:200]}...")

if __name__ == "__main__":
    main()
