from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
import faiss
import numpy as np

# ✅ Step 1: Initialize Vertex AI SDK
vertexai.init(project="" location="")

# ✅ Step 2: Sample complaints (short texts)
complaints = [
    "Battery drains quickly after charging Product A.",
    "Product B becomes very hot after a short usage.",
    "Display flickers on Product C intermittently.",
    "Charging time is too long for Product A.",
    "Product B’s camera does not autofocus properly.",
]

# ✅ Step 3: Get embeddings using vertexai module
def get_embeddings(texts):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings(texts)
    return np.array([e.values for e in embeddings]).astype("float32")

# ✅ Step 4: Embed and build FAISS index
embeddings = get_embeddings(complaints)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ✅ Step 5: Search similar complaints
def search_similar_complaints(query, k=3):
    query_embedding = get_embeddings([query])
    D, I = index.search(query_embedding, k)
    return [complaints[i] for i in I[0]]

# ✅ Step 6: Generate response using TextGenerationModel
def generate_rag_response(query):
    context = search_similar_complaints(query)
    context_block = "\n".join(context)
    prompt = f"""Context:\n{context_block}\n\nQuestion:\n{query}\n\nAnswer:"""

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(prompt=prompt, temperature=0.3, max_output_tokens=256)
    return response.text

# ✅ Example
query = "Why does my product overheat?"
answer = generate_rag_response(query)

print("🧠 Query:", query)
print("💬 RAG Answer:\n", answer)