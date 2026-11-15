import os
import streamlit as st
from sentence_transformers import SentenceTransformer

from document_loader import load_folder
from vector_store import build_index, MODEL_NAME
from rag_pipeline import RAGPipeline
from memory import ChatMemory

RAW = "data/raw_documents"
os.makedirs(RAW, exist_ok=True)

st.set_page_config(page_title="DocuMind AI", page_icon="📄")
st.title("📄 DocuMind AI")
st.caption("Chat with your PDF documents. Answers come only from the documents, with source and page.")


@st.cache_resource
def get_pipeline(config, top_k):
    return RAGPipeline(config, top_k=top_k)


if "memory" not in st.session_state:
    st.session_state.memory = ChatMemory()
memory = st.session_state.memory

# Sidebar: settings
st.sidebar.header("Settings")
config = st.sidebar.selectbox("Retrieval config", ["config1", "config2"], help="config1: 500/100, config2: 1000/150")
top_k = st.sidebar.selectbox("Top-K", [3, 5])
show_chunks = st.sidebar.checkbox("Show retrieved chunks (debug)", value=True)

# Upload and process
st.subheader("1. Documents")
uploaded = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        with open(os.path.join(RAW, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success(f"{len(uploaded)} file(s) saved.")

docs = sorted(d for d in os.listdir(RAW) if d.lower().endswith(".pdf"))
st.write("Uploaded:", ", ".join(docs) if docs else "none yet")

if st.button("Process Documents"):
    with st.spinner("Extracting, chunking and indexing..."):
        pages = load_folder(RAW)
        model = SentenceTransformer(MODEL_NAME)
        build_index(model, pages, 500, 100, "config1")
        build_index(model, pages, 1000, 150, "config2")
        st.cache_resource.clear()
    st.success(f"Indexed {len(pages)} pages from {len(docs)} documents.")

# Ask
st.subheader("2. Ask a question")
scope = st.selectbox("Search in", ["All documents"] + docs)
question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("Searching and generating..."):
        pipe = get_pipeline(config, top_k)
        doc = None if scope == "All documents" else scope
        out = pipe.ask(question.strip(), history=memory.recent(3), document=doc)
    memory.add(question.strip(), out["answer"], out["sources"])

    st.markdown("**Answer:**")
    st.write(out["answer"])
    if out["sources"]:
        st.markdown("**Sources:**")
        for s in out["sources"]:
            st.write(f"- {s['document']} - Page {s['page']}")
    if show_chunks and out["chunks"]:
        with st.expander("Retrieved chunks (debug)"):
            for c in out["chunks"]:
                st.write(f"**{c['document']} | page {c['page']} | score {c['score']}**")
                st.write(c["text"])

# History
st.subheader("3. Conversation history")
if st.button("Clear history"):
    memory.clear()
    st.rerun()
for item in reversed(memory.items):
    st.markdown(f"**Q:** {item['question']}")
    st.markdown(f"**A:** {item['answer']}")
    st.divider()