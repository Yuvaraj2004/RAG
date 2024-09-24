from IPython.display import clear_output
clear_output()

# Import libraries
import os
import threading
import subprocess
import streamlit as st
import PyPDF2
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.model_client import ModelClient
from lightrag.components.model_client import OllamaClient

# Initialize the Ollama server if not already running
def ollama():
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])

# Check if the Ollama server is already running
def is_ollama_running():
    try:
        response = subprocess.check_output(["lsof", "-i:11434"])
        return True
    except subprocess.CalledProcessError:
        return False

# Start the Ollama server if it's not already running
if not is_ollama_running():
    ollama_thread = threading.Thread(target=ollama)
    ollama_thread.start()

# Define RAG-related classes and methods
class PDFProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def extract_text_from_pdf(self, file_obj):
        # Open the PDF file using PyMuPDF
        doc = fitz.open(stream=file_obj.read(), filetype="pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)  # Use load_page instead of indexing
            text += page.get_text("text")
        return text

    def chunk_text(self, text, chunk_size=500):
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def create_embeddings(self, chunks):
        # Encode the text chunks and convert to a CPU NumPy array of type float32
        embeddings = self.model.encode(chunks, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy().astype('float32')
        return embeddings

class FAISSIndexer:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        # Ensure embeddings are in NumPy float32 format before adding to index
        if isinstance(embeddings, np.ndarray) and embeddings.dtype == 'float32':
            self.index.add(embeddings)
        else:
            raise ValueError("Embeddings must be a NumPy array of type float32.")

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return indices.flatten()

# Helper function to handle query embedding conversion
def get_query_embedding(embedding_model, query):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Check if it's a tensor and move to CPU if necessary
    if hasattr(query_embedding, 'cpu'):
        query_embedding = query_embedding.cpu().numpy().astype('float32')
    else:
        # It's already a numpy array
        query_embedding = query_embedding.astype('float32')
        
    return query_embedding

# Set up Streamlit UI
st.set_page_config(page_title="Document QA with LLaMA", layout="wide")
st.title("Document QA with LLaMA 3.1")
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    pdf_processor = PDFProcessor()
    pdf_text = pdf_processor.extract_text_from_pdf(uploaded_file)
    text_chunks = pdf_processor.chunk_text(pdf_text)
    st.write(f"Extracted {len(text_chunks)} chunks from the document.")

    embeddings = pdf_processor.create_embeddings(text_chunks)
    indexer = FAISSIndexer(embedding_dim=embeddings.shape[1])
    indexer.add_embeddings(embeddings)

    query = st.text_input("Ask a question based on the document:")
    if query:
        query_embedding = get_query_embedding(pdf_processor.model, query)
        top_chunks_indices = indexer.search(query_embedding, top_k=3)
        st.write("Top relevant chunks:")
        for idx in top_chunks_indices:
            st.write(text_chunks[idx])

            # Now use the LLaMA model to generate the answer
            model_client = OllamaClient()
            model_kwargs = {"model": "llama3.1:8b"}
            qa_template = r"""<SYS>
You are a helpful assistant.
</SYS>
User: {{input_str}}
You:"""
            generator = Generator(
                model_client=model_client,
                model_kwargs=model_kwargs,
                template=qa_template,
            )

            answer = generator.call({"input_str": text_chunks[idx] + query})
            st.write(f"Answer: {answer}")

# Allow creating, deleting, and managing chat sessions
st.sidebar.markdown("### 💬 **Chat Sessions**")
chat_sessions = []
if 'chat_sessions' not in st.session_state:
    st.session_state['chat_sessions'] = []

# New chat button
if st.sidebar.button("🆕 New Chat"):
    st.session_state['chat_sessions'].append({"chat_id": len(st.session_state['chat_sessions']) + 1, "messages": []})
    st.experimental_rerun()

# Delete chat button
if st.sidebar.button("🗑️ Delete Chat"):
    if len(st.session_state['chat_sessions']) > 0:
        st.session_state['chat_sessions'].pop()
    st.experimental_rerun()

# Display chat sessions
for chat in st.session_state['chat_sessions']:
    st.sidebar.markdown(f"**Chat ID:** {chat['chat_id']}")
    for msg in chat['messages']:
        st.sidebar.write(msg)
