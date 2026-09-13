from sentence_transformers import SentenceTransformer, util


# Original documents
corpus = [
    "Python is a programming language.",
    "Docker is used for containerization.",
    "Kubernetes manages containers."
]


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Convert documents into embeddings
corpus_embeddings = model.encode(
    corpus,
    convert_to_tensor=True
)


# User query
query = "Which technology manages containers?"


# Convert query into embedding
query_embedding = model.encode(
    query,
    convert_to_tensor=True
)


# Compare query with all documents
cosine_scores = util.cos_sim(
    query_embedding,
    corpus_embeddings
)


print("Corpus embeddings shape:")
print(corpus_embeddings.shape)

print("\nCosine scores:")
print(cosine_scores)