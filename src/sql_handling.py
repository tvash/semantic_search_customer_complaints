import os
import sqlite3
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# DB setup
conn = sqlite3.connect("db/complaints.db")
batch_size = 1000
offset = 0

vectorstore = None  # will hold the full FAISS index

print("📦 Starting embedding and indexing...")

while True:
    # Load batch of complaints
    query = f"SELECT * FROM complaints LIMIT {batch_size} OFFSET {offset}"
    df = pd.read_sql_query(query, conn)
    if df.empty:
        break
    offset += batch_size

    print(f"🔄 Processing batch {offset // batch_size}")

    docs = []
    for _, row in df.iterrows():
        metadata = {
            "Complaint ID": row["Complaint ID"],
            "Product": row["Product"],
            "Issue Type": row["Issue Type"],
            "Amount Involved": str(row["Amount Involved"]),
            "Customer Segment": row["Customer Segment"],
            "Channel": row["Channel"],
            "Transaction Date": row["Transaction Date"],
            "Priority": row["Priority"],
        }
        doc = Document(page_content=row["Complaint Text"], metadata=metadata)
        split_docs = text_splitter.split_documents([doc])
        docs.extend(split_docs)

    # Embed and add to vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embedding_model)
    else:
        vectorstore.add_documents(docs)

# Save FAISS index to disk
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore/")
print("✅ Vectorstore saved to disk.")