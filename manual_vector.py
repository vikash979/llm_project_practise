import numpy as np
import faiss

# Existing document embeddings
documents = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [10.0, 10.0]
], dtype="float32")

# Create FAISS index
index = faiss.IndexFlatL2(2)

# Add vectors
index.add(documents)

# User query vector
query = np.array([
    [1.2, 1.8]
], dtype="float32")

# Search top 2
distances, indexes = index.search(query, 2)

print("Distances:", distances)
print("Indexes:", indexes)