import streamlit as st
from rag_chatbot import *
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# UI Configuration
st.set_page_config(page_title="🧠 RAG Pro", layout="wide")
st.title("🧠 RAG Pro - Advanced Document Assistant")

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    use_reranker = st.toggle("Enable Reranker", True)
    mode = st.radio("Mode:", ["Q&A", "Summarization"])
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
    
    if st.button("Reload Documents"):
        st.session_state.docs_loaded = False
        st.session_state.vectorstore = None
        st.rerun()

# Document Loading
if not st.session_state.docs_loaded:
    with st.status("Processing documents...", expanded=True) as status:
        st.write("Loading documents...")
        raw_docs = load_documents("docs")
        
        st.write("Chunking documents...")
        chunked_docs = chunk_documents(raw_docs, chunk_size, chunk_overlap)
        
        st.write("Creating vector store...")
        st.session_state.vectorstore = create_vectorstore(chunked_docs)
        
        st.write("Initializing QA system...")
        st.session_state.rag_chain = build_rag_chain(
            st.session_state.vectorstore, 
            use_reranker
        )
        st.session_state.docs_loaded = True
        status.update(label="Processing complete!", state="complete")

# Chat Interface
st.subheader("Document Interaction")
query = st.chat_input("Ask about your documents...")

if query:
    # Add to history
    st.session_state.history.append({"role": "user", "content": query})
    
    with st.spinner("Thinking..."):
        if mode == "Q&A":
            response = st.session_state.rag_chain.invoke(query)
            answer = response["result"]
            sources = response["source_documents"]
            
            # Add to history
            st.session_state.history.append({"role": "assistant", "content": answer})
        else:
            answer = summarize_documents(load_documents("docs"))
            st.session_state.history.append({"role": "assistant", "content": answer.content})

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Display sources if in Q&A mode
if query and mode == "Q&A" and "sources" in locals():
    with st.expander("Source Documents"):
        for i, source in enumerate(sources):
            st.caption(f"Source {i+1}: {source.metadata.get('source', 'Unknown')}")
            st.text(source.page_content[:500] + "...")
            st.divider()

# Document Info
if st.session_state.docs_loaded:
    with st.sidebar:
        st.divider()
        st.subheader("Document Status")
        st.success(f"Documents processed: {len(load_documents('docs'))}")
        st.info(f"Chunks created: {len(st.session_state.vectorstore.index_to_docstore_id)}")
        st.caption(f"Embedding model: text-embedding-3-small")