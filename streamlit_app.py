import streamlit as st
from app import chat

st.set_page_config(
    page_title="Customer Support RAG Bot",
    page_icon="🤖"
)

st.title("🤖 Context-Aware Customer Support RAG Bot")

st.write("Enter your User ID and ask a question.")

user_id = st.number_input(
    "User ID",
    min_value=1,
    step=1
)

question = st.text_area(
    "Question"
)

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating answer..."):

            answer = chat(int(user_id), question)

        st.success(answer)