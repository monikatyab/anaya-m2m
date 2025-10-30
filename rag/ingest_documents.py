"""
Anaya AI - Document Ingestion Script

Instructions:
1. Add PDF, DOCX, or TXT files to the 'knowledge_base/' folder
2. Run: python ingest_documents.py
3. Done! Your documents are now in Anaya's knowledge base.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_rag import ingest_all_documents

if __name__ == "__main__":
    print("\n🌾 Welcome to Anaya AI Knowledge Base Ingestion\n")
    print("This will process all documents in the knowledge_base/ folder.")
    print("Make sure you've added your PDF, DOCX, or TXT files there first!\n")
    
    # Run the ingestion
    ingest_all_documents()
    
    print("\n All done! Anaya can now access your knowledge base.")
    print("💬 Start chatting: python main.py\n")
