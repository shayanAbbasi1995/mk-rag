import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

st.set_page_config(page_title="Course Q&A Assistant", layout="centered")
st.title("📚 Course Materials Assistant")

# Secrets from environment or Streamlit Secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPEN_AI_EMBEDDINGS_API_KEY", os.getenv("OPEN_AI_EMBEDDINGS_API_KEY"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))


@st.cache_resource
def get_retriever():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="course-docs",
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": 4})


retriever = get_retriever()

llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the course material..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            docs = retriever.invoke(prompt)
            context = "\n\n".join(
                [f"[{d.metadata.get('source_file', 'Unknown')}]: {d.page_content}" for d in docs]
            )
            sources = list(set([d.metadata.get("source_file", "Unknown") for d in docs]))

            system_prompt = (
                "You are a helpful teaching assistant. Answer the student's question strictly "
                "using the context below. If the answer cannot be found in the context, say "
                "you do not know based on the provided material. Do not fabricate answers.\n\n"
                f"Context:\n{context}"
            )

            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ])

            reply_content = response.content

            if sources:
                reply_content += f"\n\n*Referenced Material: {', '.join(sources)}*"

            st.markdown(reply_content)
            st.session_state.messages.append({"role": "assistant", "content": reply_content})

        except Exception as e:
            st.error(f"Could not generate a response: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"[Error: {e}]"})
