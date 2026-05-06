import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Path Setup
DATA_PATH = "data/"
DB_PATH = "chroma_db"

def ingest_docs():
    # STEP 1: Load all PDFs from the data/ folder
    print(f"📂 Loading PDFs from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"✅ Loaded {len(raw_documents)} pages.")

    # STEP 2: Split text into chunks 
    # (Important for LLMs to handle long documents)
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(docs)} text chunks.")

    # STEP 3: Create Embeddings
    print("🧠 Generating embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # STEP 4: Store in ChromaDB
    print(f"💾 Saving to {DB_PATH}...")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("🏁 Ingestion Complete! Your Automotive AI is ready.")

if __name__ == "__main__":
    ingest_docs()