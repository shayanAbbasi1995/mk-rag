"""
Student-facing chat interface for the course RAG assistant.

Run: streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "course-docs"
TOP_K = 5

st.set_page_config(page_title="Course Assistant", page_icon="📚")
st.title("📚 Course Assistant")
st.caption("Ask questions about your course materials.")


@st.cache_resource
def build_chain():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPEN_AI_EMBEDDINGS_API_KEY"),
    )
    qdrant = QdrantVectorStore(
        client=QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        ),
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPEN_AI_EMBEDDINGS_API_KEY"),
        temperature=0,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=qdrant.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,
    )


chain = build_chain()

if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if prompt := st.chat_input("Ask a question about the course..."):
    st.chat_message("user").write(prompt)

    result = chain.invoke(
        {"question": prompt, "chat_history": st.session_state.chat_history}
    )
    answer = result["answer"]
    sources = {doc.metadata.get("source_file", "unknown") for doc in result["source_documents"]}

    st.chat_message("assistant").write(answer)
    if sources:
        st.caption(f"Sources: {', '.join(sorted(sources))}")

    st.session_state.history.extend([("user", prompt), ("assistant", answer)])
    st.session_state.chat_history.append((prompt, answer))
