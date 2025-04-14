# RAG Fusion from CSV - Full Working Example with Reranking

import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load complaints from CSV
df = pd.read_csv("data/customer_complaints.csv")

# Step 2: Create LangChain documents with metadata
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
        "source": f"Complaint ID: {row.get('Complaint ID', '')}"  # 👈 Required fix

    }
    text = row.get("Complaint Text", "")
    doc = Document(page_content=text, metadata=metadata)
    documents.append(doc)

# Step 3: Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Step 4: Create embeddings and vectorstore
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 5: Save the vectorstore
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore/")
print("\n✅ Vectorstore created and saved successfully.")

# Step 6: Load vectorstore with safe deserialization
db = FAISS.load_local("vectorstore/", embedding_model, allow_dangerous_deserialization=True)

# Step 7: Use MultiQueryRetriever for RAG Fusion
llm = OpenAI(openai_api_key=openai_api_key)
retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(search_kwargs={"k": 8}),
    llm=llm
)

# Step 8: Retrieval QA Chain with Source Documents
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Step 9: Interactive Q&A
while True:
    question = input("\n💬 Ask a question (or type 'exit'): ")
    if question.lower() == "exit":
        break

    result = qa_chain({"question": question})
    print("\n🤖 Answer:", result["answer"])

    print("\n📄 Source Documents:")
    for doc in result["source_documents"]:
        print(f" - Complaint ID: {doc.metadata['Complaint ID']} | Product: {doc.metadata['Product']}")

