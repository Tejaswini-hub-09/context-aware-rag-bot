import os
import sqlite3
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception(
        "GROQ_API_KEY not found. Please set it in .env or Streamlit Secrets."
    )

# ==========================
# Initialize Groq LLM
# ==========================
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=GROQ_API_KEY,
    temperature=0
)

# ==========================
# Embeddings
# ==========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Load Chroma
# ==========================
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# ==========================
# SQLite
# ==========================
def get_user(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, membership_tier FROM users WHERE user_id=?",
        (user_id,)
    )

    user = cursor.fetchone()
    conn.close()

    return user


# ==========================
# Chat Function
# ==========================
def chat(user_id, question):

    user = get_user(user_id)

    if user is None:
        return "User not found. Please enter a valid user_id."

    name, membership = user

    docs = vectorstore.similarity_search(question, k=3)

    if len(docs) == 0:
        return "I do not have enough information in the provided knowledge base to answer this."

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an AI customer support assistant.

You are speaking with:

Name: {name}
Membership Tier: {membership}

Answer the user's question using ONLY the context below.

If the answer is not available, reply exactly:

"I do not have enough information in the provided knowledge base to answer this."

Context:
{context}

User Question:
{question}

Answer:
"""

    try:
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Error communicating with Groq API:\n{e}"


# ==========================
# Terminal Chatbot
# ==========================
if __name__ == "__main__":

    print("=" * 60)
    print("AI CUSTOMER SUPPORT RAG BOT")
    print("=" * 60)

    while True:

        uid = input("\nEnter User ID (or exit): ")

        if uid.lower() == "exit":
            break

        if not uid.isdigit():
            print("Please enter a valid numeric user ID.")
            continue

        question = input("Ask your question: ")

        answer = chat(int(uid), question)

        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)