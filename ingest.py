from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
import shutil

# Load FAQ document
loader = TextLoader("company_faq.txt", encoding="utf-8")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Remove existing Chroma DB if present
persist_directory = "chroma_db"

if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)

print("✅ Knowledge Base Created Successfully!")