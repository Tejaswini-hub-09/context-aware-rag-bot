import os
import sqlite3

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==============================
# Load Environment Variables
# ==============================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY not found. Please create a .env file.")

# ==============================
# Initialize Groq Model
# ==============================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0
)

# ==============================
# Load Embedding Model
# ==============================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# Load Chroma Vector Database
# ==============================

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# ==============================
# FastAPI App
# ==============================

app = FastAPI(
    title="Context Aware Customer Support RAG Bot",
    version="1.0"
)

# ==============================
# Home Page
# ==============================

@app.get("/")
def home():
    return {
        "message": "Context Aware Customer Support RAG Bot is Running!",
        "swagger_docs": "http://127.0.0.1:8000/docs"
    }

# ==============================
# Request Model
# ==============================

class ChatRequest(BaseModel):
    user_id: int
    user_query: str

# ==============================
# SQLite Function
# ==============================

def get_user(user_id):

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, membership_tier FROM users WHERE user_id=?",
        (user_id,)
    )

    row = cursor.fetchone()

    conn.close()

    return row

# ==============================
# Chat Endpoint
# ==============================

@app.post("/chat")
def chat(request: ChatRequest):

    user = get_user(request.user_id)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found. Please enter a valid user_id."
        )

    docs = vectorstore.similarity_search(
        request.user_query,
        k=5
    )

    if len(docs) == 0:
        return {
            "answer": "I do not have enough information in the provided knowledge base to answer this."
        }

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are an AI customer support assistant.

You are speaking with:

Name: {user[0]}

Membership Tier: {user[1]}

Answer the user's question using ONLY the context below.

If the answer is not available in the context, say exactly:

"I do not have enough information in the provided knowledge base to answer this."

Context:
{context}

User Question:
{request.user_query}

Answer:
"""

    try:

        response = llm.invoke(prompt)

        return {
            "user_id": request.user_id,
            "name": user[0],
            "membership_tier": user[1],
            "question": request.user_query,
            "answer": response.content
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )