
#LANGCHAIN LIBRARIES
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

#READING OTHER LIBRARIES
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken

#TODO: Create MAP function to store in document

# Load OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Read CSV data
df = pd.read_csv("data/customer_complaints.csv")

# Create documents with metadata
documents = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    metadata = {
        "Complaint ID": row.get("Complaint ID", ""),
        "Product": row.get("Product", ""),
        "Issue Type": row.get("Issue Type", ""),
        "Complaint Text": row.get("Complaint Text", ""),
        "Amount Involved": str(row.get("Amount Involved", "")),
        "Customer Segment": row.get("Customer Segment", ""),
        "Channel": row.get("Channel", ""),
        "Transaction Date": row.get("Transaction Date", ""),
        "Priority": row.get("Priority", ""),
    }
    text = row.get("Complaint Text", "")
    print("\n text:\n", text)
    doc = Document(page_content=text, metadata=metadata)
    documents.append(doc)

# Split long texts
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

print(docs)

# Build vectorstore
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save to disk
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore/")
print("✅ Vectorstore created and saved.")