import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util


# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [
    "Python is a programming language used for backend development.",
    "Docker is a containerization platform used to package applications.",
    "Kubernetes is used to deploy, manage, and scale containers.",
    "Django is a Python web framework.",
    "FastAPI is a modern Python framework for building APIs."
]


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 3. CREATE DOCUMENT EMBEDDINGS
# ============================================================

document_embeddings = model.encode(
    documents,
    convert_to_numpy=True
)

# FAISS requires float32
document_embeddings = document_embeddings.astype("float32")

print("Embedding shape:")
print(document_embeddings.shape)


# ============================================================
# 4. NORMALIZE DOCUMENT EMBEDDINGS
# ============================================================

faiss.normalize_L2(document_embeddings)


# ============================================================
# 5. CREATE FAISS INDEX
# ============================================================

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)


# ============================================================
# 6. ADD DOCUMENT VECTORS
# ============================================================

index.add(document_embeddings)

print("\nVectors stored in FAISS:")
print(index.ntotal)


# ============================================================
# 7. USER QUERY
# ============================================================

query = "Which technology manages containers?"


# ============================================================
# 8. CREATE QUERY EMBEDDING
# ============================================================

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

query_embedding = query_embedding.astype("float32")


# ============================================================
# 9. NORMALIZE QUERY EMBEDDING
# ============================================================

faiss.normalize_L2(query_embedding)


# ============================================================
# 10. SEARCH TOP-K SIMILAR DOCUMENTS
# ============================================================

top_k = 3

similarities, indices = index.search(
    query_embedding,
    top_k
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\nUser Query:")
print(query)

print("\nSearch Results:")

for rank, (idx, similarity) in enumerate(
    zip(indices[0], similarities[0]),
    start=1
):
    print(f"\nRank {rank}")
    print("Cosine Similarity:", similarity)
    print("Document:", documents[idx])


# similarity = util.cos_sim(
#     query_embedding[0],
#     query_embedding[1]
# )
similarity = util.cos_sim(
    query_embedding,
    document_embeddings
)
# sentences = [
#     "Python is a programming language.",
#     "Python is used for software development.",
#     "I like eating pizza."
# ]

# embeddings = model.encode(sentences)
print("===============calculating cosine similarity directly.====", similarity)