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
    print("❌ GROQ_API_KEY not found.")
    print("Please create a .env file and add:")
    print("GROQ_API_KEY=your_groq_api_key")
    exit()

# ==========================
# Initialize Groq LLM
# ==========================
try:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # Updated supported model
        api_key=GROQ_API_KEY,
        temperature=0
    )
except Exception as e:
    print("Error initializing Groq:")
    print(e)
    exit()

# ==========================
# Load Embedding Model
# ==========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Load Chroma Vector Store
# ==========================
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# ==========================
# SQLite Function
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
# Chatbot
# ==========================
print("=" * 60)
print("      AI CUSTOMER SUPPORT RAG CHATBOT")
print("=" * 60)

while True:

    user_input = input("\nEnter User ID (or 'exit'): ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not user_input.isdigit():
        print("Please enter a valid numeric user ID.")
        continue

    user = get_user(int(user_input))

    if user is None:
        print("User not found. Please enter a valid user_id.")
        continue

    name = user[0]
    membership = user[1]

    print(f"\nWelcome, {name}")
    print(f"Membership Tier: {membership}")

    question = input("\nAsk your question: ")

    if question.lower() == "exit":
        break

    # ==========================
    # Retrieve Relevant Chunks
    # ==========================
    docs = vectorstore.similarity_search(question, k=3)

    if len(docs) == 0:
        print("I do not have enough information in the provided knowledge base to answer this.")
        continue

    context = "\n\n".join(doc.page_content for doc in docs)

    # ==========================
    # Prompt
    # ==========================
    prompt = f"""
You are an AI customer support assistant.

You are speaking with:

Name: {name}
Membership Tier: {membership}

Answer the user's question using ONLY the context provided below.

If the answer is not available in the context, reply exactly:

"I do not have enough information in the provided knowledge base to answer this."

Context:
{context}

User Question:
{question}

Answer:
"""

    # ==========================
    # Generate Response
    # ==========================
    try:
        response = llm.invoke(prompt)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(response.content)

    except Exception as e:
        print("\nError communicating with Groq API.")
        print(e)