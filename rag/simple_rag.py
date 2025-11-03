"""
Simple RAG System for Anaya AI
Handles document ingestion and retrieval
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = "./rag/knowledge_base"
VECTOR_DB_PATH = "./rag/anaya_knowledge_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 6

# Embedding model
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",   
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Load all documents from knowledge base folder"""
    documents = []
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    print(f"\n Scanning {folder_path} for documents...")
    
    # Load PDFs
    try:
        pdf_loader = DirectoryLoader(
            folder_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"Warning: Error loading PDFs: {e}")
    
    # Load DOCX files
    try:
        docx_loader = DirectoryLoader(
            folder_path,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True
        )
        documents.extend(docx_loader.load())
    except Exception as e:
        print(f"Warning: Error loading DOCX files: {e}")
    
    # Load TXT files
    try:
        txt_loader = DirectoryLoader(
            folder_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        documents.extend(txt_loader.load())
    except Exception as e:
        print(f"Warning: Error loading TXT files: {e}")
    
    print(f" Loaded {len(documents)} document chunks")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller, overlapping chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"📝 Split into {len(chunks)} chunks")
    return chunks


def create_or_update_vectordb(documents: List[Document]):
    """Create or update the vector database"""
    if not documents:
        print(" No documents to index!")
        return
    
    print(f"\n🔧 Creating/updating vector database at {VECTOR_DB_PATH}...")
    
    # Remove old database if exists
    if os.path.exists(VECTOR_DB_PATH):
        print(" Removing old database...")
        import shutil
        shutil.rmtree(VECTOR_DB_PATH)
    
    # Create new vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
        persist_directory=VECTOR_DB_PATH,
        collection_name="anaya-knowledge"
    )
    
    print(f" Vector database created with {len(documents)} chunks!")
    return vectorstore


def get_retriever():
    """Get retriever for querying the knowledge base"""
    if not os.path.exists(VECTOR_DB_PATH):
        print(" No vector database found! Run: python -m rag.ingest_documents")
        return None
    
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embed_model,
        collection_name="anaya-knowledge"
    )
    
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_RESULTS, "fetch_k": 20}
    )


def ingest_all_documents():
    """Main ingestion function"""
    print("\n" + "="*60)
    print(" ANAYA AI - KNOWLEDGE BASE INGESTION")
    print("="*60)
    
    # Load documents
    documents = load_documents_from_folder(KNOWLEDGE_BASE_DIR)
    
    if not documents:
        print("\nNo documents found in knowledge_base/ folder!")
        print("Add PDF, DOCX, or TXT files to knowledge_base/ and run again.")
        return
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    # Create vector database
    create_or_update_vectordb(chunks)
    
    print("\n" + "="*60)
    print("✨ INGESTION COMPLETE!")
    print("="*60)
    print(f" Total documents processed: {len(documents)}")
    print(f" Total chunks created: {len(chunks)}")
    print(f" Database location: {VECTOR_DB_PATH}")
    print("\n The knowledge base is now ready for use by Anaya AI agents!")
