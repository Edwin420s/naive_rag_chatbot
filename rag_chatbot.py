import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts import PromptTemplate
from utils import log_retrieval  # Custom logging

# ---------------------------
# Document Loading & Chunking
# ---------------------------
def load_documents(doc_folder="docs"):
    docs = []
    for file_name in os.listdir(doc_folder):
        file_path = os.path.join(doc_folder, file_name)
        try:
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                loader = UnstructuredFileLoader(file_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(docs)

# ---------------------------
# Vector Store
# ---------------------------
def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore")  # Persistent storage
    return vectorstore

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# ---------------------------
# RAG Chain with Reranking
# ---------------------------
def build_rag_chain(vectorstore, use_reranker=False):
    prompt_template = """Use the following context to answer the question. 
If you don't know the answer, say you don't know. Keep answers concise.

Context:
{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10 if use_reranker else 3}
    )

    if use_reranker:
        compressor = CohereRerank(top_n=3)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    else:
        retriever = base_retriever

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

# ---------------------------
# Summarization Mode
# ---------------------------
def summarize_documents(docs):
    summary_prompt = """
Summarize the key points from these documents in 3-5 bullet points. 
Focus on main themes, important conclusions, and actionable insights.

Documents:
{text}
"""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    return llm.invoke(summary_prompt.format(text="\n\n".join([d.page_content for d in docs])))

# ---------------------------
# Security / Sanitization
# ---------------------------
def sanitize_input(query):
    return re.sub(r"[^a-zA-Z0-9\s.,?\-]", "", query)
