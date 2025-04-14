import os
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load complaint data
df = pd.read_csv("data/complaints.csv")

# Convert each row into a Document
documents = []
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
    documents.append(Document(page_content=row["Complaint Text"], metadata=metadata))

# Optional: Chunk large texts (if needed)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create vector store with OpenAI embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embedding_model)

# Create retriever and QA chain
retriever = db.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key, temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Ask questions
while True:
    query = input("\n💬 Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain(query)
    print("\n🤖 Answer:", result["result"])
    
    print("\n📄 Top Sources:")
    for doc in result["source_documents"]:
        print(f"  - Complaint ID: {doc.metadata['Complaint ID']} | Product: {doc.metadata['Product']}")