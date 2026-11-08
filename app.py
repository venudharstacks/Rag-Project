
import os
import tempfile
import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with Your PDF")
st.write("Upload a PDF and ask questions about its content.")

# Gemini API setup
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Please configure your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Text chunking
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# PDF extraction
def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages

# Build FAISS index
def build_index(chunks):
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index

# Retrieve relevant chunks
def retrieve(question, index, chunks, k=4):
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(
        query_embedding,
        min(k, len(chunks))
    )

    return [
        chunks[i]
        for i in indices[0]
        if i >= 0
    ]

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:
    with st.spinner("Reading PDF..."):
        pages = extract_pdf(uploaded_file)

    if not pages:
        st.error("No readable text found in this PDF.")
        st.stop()

    # Preserve page information in each chunk
    chunks = []

    for page in pages:
        page_chunks = split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "page": page["page"],
                "text": chunk
            })

    if not chunks:
        st.error("Could not create text chunks.")
        st.stop()

    with st.spinner("Creating FAISS vector database..."):
        chunk_texts = [item["text"] for item in chunks]
        index = build_index(chunk_texts)

    st.success(f"PDF processed! Created {len(chunks)} chunks.")

    question = st.text_input("Ask a question about your PDF:")

    if st.button("Get Answer") and question.strip():
        with st.spinner("Searching PDF and generating answer..."):
            relevant_chunks = retrieve(
                question,
                index,
                chunks
            )

            context = "\n\n".join(
                f"Page {item['page']}:\n{item['text']}"
                for item in relevant_chunks
            )

            prompt = f"""
You are a helpful PDF question-answering assistant.

Answer using ONLY the provided PDF context.
If the answer is not present, say you could not find it in the PDF.
Mention the relevant page numbers in your answer.

PDF CONTEXT:
{context}

USER QUESTION:
{question}
"""

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

        st.subheader("Answer")
        st.write(response.text)

        with st.expander("View retrieved PDF context"):
            st.write(context)
