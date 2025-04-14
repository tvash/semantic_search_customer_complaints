import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from google.cloud import aiplatform

# Init Vertex AI
aiplatform.init(project="your-gcp-project-id", location="us-central1")

# Init Weaviate client (local or remote)
client = weaviate.Client("http://localhost:8080")  # Use WCS URL if cloud

# Embedding model
embedding = VertexAIEmbeddings(model_name="textembedding-gecko@latest")


complaints = [
    "I was charged twice on my credit card.",
    "My internet disconnects every night around 10 PM.",
    "App keeps crashing when I try to upload a photo.",
    "My refund hasn't been processed even after 10 days.",
    "Wi-Fi is slow in the evenings, can't stream videos.",
]

docs = [Document(page_content=txt) for txt in complaints]

# Optional: Reset Weaviate schema
if client.schema.exists("CustomerComplaint"):
    client.schema.delete_class("CustomerComplaint")

# Define schema for Weaviate
schema = {
    "class": "CustomerComplaint",
    "vectorizer": "none",  # We’ll provide embeddings manually
    "properties": [{"name": "text", "dataType": ["text"]}],
}
client.schema.create_class(schema)

# Create LangChain Weaviate vector store
vectorstore = Weaviate(
    client=client,
    index_name="CustomerComplaint",
    text_key="text",
    embedding=embedding,
)

# Upload documents
vectorstore.add_documents(docs)


# Load Gemini via LangChain
llm = VertexAI(model_name="gemini-pro")

# Create RAG chain using Retriever + Gemini
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Ask a question
query = "Why is my internet slow every night?"
response = qa_chain(query)

print("🤖 Gemini Response:\n", response["result"])
print("\n📚 Retrieved Complaints:")
for doc in response["source_documents"]:
    print("-", doc.page_content)

# 🤖 Gemini Response:
# Many users report internet slowdowns during the evening, likely due to peak-hour congestion. It’s a common service issue, especially with shared bandwidth.

# 📚 Retrieved Complaints:
# - My internet disconnects every night around 10 PM.
# - Wi-Fi is slow in the evenings, can't stream videos.

