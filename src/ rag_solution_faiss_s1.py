# rag_solution.py
#Solution using FAISS
import os
import numpy as np
import pandas as pd
from google.cloud import aiplatform
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA

# Set your Google Cloud project details (replace with your values)
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ---------- Data Ingestion ----------
# For demo purposes, we define some sample customer complaints.
complaints = [
    "I was charged twice on my credit card.",
    "My internet disconnects every night around 10 PM.",
    "The app crashes every time I try to upload a photo.",
    "My refund hasn't been processed even after 10 days.",
    "Wi-Fi is slow in the evenings, and I can't stream videos properly."
]

# Convert complaints into LangChain Document objects
docs = [Document(page_content=text) for text in complaints]

# ---------- Create Vector Store (FAISS) ----------
# Initialize the Vertex AI embedding model.
embedding_model = VertexAIEmbeddings(model_name="textembedding-gecko@latest")
# Build the FAISS index from documents.
vectorstore = FAISS.from_documents(docs, embedding_model)

# ---------- Build RAG Chain ----------
# Initialize the Vertex AI Gemini model for generation.
llm = VertexAI(model_name="gemini-pro")
# Create a RetrievalQA chain that retrieves relevant complaints and generates an answer.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ---------- Query the RAG System ----------
if __name__ == "__main__":
    # Sample user query
    query = "Why is my internet so slow every night?"
    response = qa_chain(query)

    # Output the generated response and the source documents
    print("📩 Query:", query)
    print("\n🤖 Gemini Response:\n", response["result"])
    print("\n📚 Retrieved Complaints:")
    for doc in response["source_documents"]:
        print("-", doc.page_content)



# +--------------------------+
# |  Complaint Source (CSV)  | <- GCS, BigQuery, Pub/Sub
# +--------------------------+
#            |
#         [Airflow]
#            ↓
# +--------------------------+
# | Embedding: Vertex AI     | <- textembedding-gecko
# +--------------------------+
#            |
#         [Python Job]
#            ↓
# +--------------------------+
# | Vector DB (Weaviate)     |
# +--------------------------+
#            ↑
#            |          +---------------------+
#            |<---------|  User Query API     |
#            |          +---------------------+
#            |          | Embed + Retrieve +  |
#            |          | RAG via Gemini Pro  |
#            |          +---------------------+
#            |
#       [Vertex AI Gemini]

# +-------------------------------+
# | Monitoring: Logs, Metrics     |
# | CI/CD: Cloud Build + GitHub   |
# | Retraining Trigger Pipelines  |
# +-------------------------------+


        