
import os
os.environ["LANGCHAIN_SAFE_SERIALIZATION"] = "false"
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv


#Cosin similarity:
import numpy as np

#COSINE SIMILARITY TO CHECKE THE DIFFERENCE
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

#GET THE NUMBER OF TOKENS 
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# num_tokens_from_string(question, "cl100k_base")

# similarity = cosine_similarity(query_result, document_result)
# print("Cosine Similarity:", similarity)

# Load key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load vectorstore
vectorstore = FAISS.load_local("vectorstore", 
                                embedding_model,
                                 allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key, temperature=0),
    retriever=retriever,
    return_source_documents=True
)

# User interface
while True:
    question = input("\n💬 Ask a question (or type 'exit'): ")
    if question.lower() == "exit":
        break

    result = qa_chain(question)
    print("\n🤖 Answer:", result["result"])
    print("\n📄 Sources:")
    for doc in result["source_documents"]:
        # print(f" - Complaint ID: {doc.metadata['Complaint ID']}, Product: {doc.metadata['Product']}, Result {result}")