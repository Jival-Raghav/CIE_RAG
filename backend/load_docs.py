import os
from pathlib import Path
from vector import QdrantManager

# Create the manager
manager = QdrantManager(collection_name="docs")

# Define the folder where your PDFs are stored
DATA_DIR = Path(__file__).parent / "data"

# Make sure the folder exists
if not DATA_DIR.exists():
    print(f"❌ Folder does not exist: {DATA_DIR}")
    exit()

# Find all PDFs in the folder
pdf_files = list(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    print(f"⚠️ No PDF files found in {DATA_DIR}")
    exit()

# Process each PDF
for pdf in pdf_files:
    print(f"📄 Processing: {pdf.name}")
    chunks = manager.process_file(str(pdf))
    
    if chunks:
        print(f"✅ Extracted {len(chunks)} chunks")
        stored = manager.store_documents(chunks)
        if stored:
            print(f"🧠 Stored to vector DB: {pdf.name}")
        else:
            print(f"⚠️ Failed to store: {pdf.name}")
    else:
        print(f"⚠️ No chunks extracted from: {pdf.name}")
