#pip install sentence-transformers faiss-cpu numpy
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [
    "Python is a high-level programming language. "
    "Python is widely used for backend development, "
    "data science, machine learning, automation, and scripting.",

    "Docker is a containerization platform. "
    "It packages applications and their dependencies into containers. "
    "Docker is commonly used in CI/CD pipelines and microservices.",

    "Kubernetes is a container orchestration platform. "
    "It is used to deploy, manage, and scale containerized applications. "
    "Kubernetes clusters contain nodes and pods."
]


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 3. CREATE EMBEDDINGS FOR DOCUMENTS
# ============================================================

document_embeddings = model.encode(
    documents,
    convert_to_numpy=True
)

print(f"\n \n ---------------------{document_embeddings}-----------------------------------\n")


# FAISS requires float32
document_embeddings = document_embeddings.astype("float32")

print("Embedding shape:", document_embeddings.shape)


# ============================================================
# 4. CREATE FAISS INDEX
# ============================================================

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)


# ============================================================
# 5. ADD DOCUMENT EMBEDDINGS TO FAISS
# ============================================================

index.add(document_embeddings)

print("Number of vectors in FAISS:", index.ntotal)


# ============================================================
# 6. USER QUERY
# ============================================================

query = "Which technology is used to manage containers?"


# ============================================================
# 7. CREATE EMBEDDING FOR USER QUERY
# ============================================================

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

query_embedding = query_embedding.astype("float32")


# ============================================================
# 8. SEARCH SIMILAR VECTORS
# ============================================================

top_k = 2

distances, indices = index.search(
    query_embedding,
    top_k
)


# ============================================================
# 9. DISPLAY RETRIEVED DOCUMENTS
# ============================================================

print("\nUser Query:")
print(query)

print("\nRetrieved Documents:")

for rank, (idx, distance) in enumerate(
    zip(indices[0], distances[0]),
    start=1
):
    print(f"\nRank {rank}")
    print("Distance:", distance)
    print("Document:", documents[idx])