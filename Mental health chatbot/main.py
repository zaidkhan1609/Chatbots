# === rag_chatbot_pdf/utils/pdf_loader.py ===
# Loads and extracts text from multiple PDFs in the folder

import fitz  # PyMuPDF
import os


def load_pdfs_from_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            with fitz.open(filepath) as doc:
                for page in doc:
                    all_text += page.get_text()
    return all_text


# === rag_chatbot_pdf/utils/chunk_text.py ===
# Splits long text into smaller chunks using LangChain

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=700, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ","]
    )
    return splitter.split_text(text)


# === rag_chatbot_pdf/utils/embedder.py ===
# Converts chunks into embeddings using Sentence-BERT

from sentence_transformers import SentenceTransformer
import numpy as np


def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings), chunks


# === rag_chatbot_pdf/utils/vector_store.py ===
# Stores embeddings and text chunks into a FAISS index

import faiss
import pickle


def build_faiss_index(embeddings, chunks, index_path="chunks/faiss_index.index", metadata_path="chunks/chunks.pkl"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    print("[INFO] FAISS index and chunk metadata saved.")


def load_faiss_index(index_path="chunks/faiss_index.index", metadata_path="chunks/chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# === rag_chatbot_pdf/utils/rag_chain.py ===
# Uses LangChain to answer questions from retrieved PDF chunks

from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def load_llm_pipeline(model_id="google/flan-t5-xl"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)


import os

class SimpleRAGPipeline:
    def __init__(self, model_id="google/flan-t5-xl"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # If index doesn't exist, build it
        if not os.path.exists("chunks/index.faiss"):
            print("[INFO] FAISS index not found. Creating it...")
            text = load_pdfs_from_folder("data/your_pdfs")
            chunks = chunk_text(text)
            embeddings, _ = embed_chunks(chunks)

            self.vectorstore = FAISS.from_texts(
                texts=chunks,
                embedding=self.embeddings
            )

            self.vectorstore.save_local("chunks")  # Save as index.faiss + .pkl
            print("[INFO] FAISS index created and saved.")
        else:
            self.vectorstore = FAISS.load_local(
                folder_path="chunks",
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm_pipeline(model_id),
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

    def ask(self, query):
        result = self.qa_chain({"query": query})
        return result["result"], result["source_documents"]


# === rag_chatbot_pdf/app.py ===
# Streamlit frontend for interacting with the chatbot

import streamlit as st
import os

st.set_page_config(page_title="🧠 Mental Health PDF Chatbot", layout="wide")
st.title("🧠 Mental Health PDF Chatbot")
st.write("Ask questions based on WHO mental health guidelines.")

pdf_folder = os.path.join(os.getcwd(), "data")
os.makedirs("chunks", exist_ok=True)

if not os.path.exists("chunks/faiss_index.index") or not os.path.exists("chunks/chunks.pkl"):
    with st.spinner("🔍 Processing PDFs..."):
        full_text = load_pdfs_from_folder(pdf_folder)
        chunks = chunk_text(full_text)
        embeddings, chunks = embed_chunks(chunks)
        build_faiss_index(embeddings, chunks)

index, texts = load_faiss_index()
rag_pipeline = SimpleRAGPipeline()

query = st.text_input("💬 Ask a question:")

if query:
    with st.spinner("🤖 Thinking..."):
        answer, sources = rag_pipeline.ask(query)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("📚 Source Chunks")
        for i, doc in enumerate(sources):
            st.markdown(f"**Chunk {i+1}**: {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")
