from sentence_transformers import SentenceTransformer
from scipy.spatial import distance


device = "mps"
# Load the models
# so close, and yet, .! so far ~!~ ~
providers = [
    ('CoreMLExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider',
]

model1 = SentenceTransformer('./models/optimum/all-MiniLM-L6-v2', device=device,
                             model_args={
                                    "providers": providers
                                })
print("\033[91m", model1.modules(), "\033[0m")
# model2 = SentenceTransformer('./models/all-MiniLM-L6-v2', device=device)

sentences = [
    'This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.'
]



# Get embeddings for each sentence from both models
embeddings1 = model1.encode(sentences)
# embeddings2 = model2.encode(sentences)


# Compute and print the cosine similarity for each sentence's embeddings from the two models
for sentence, emb1, emb2 in zip(sentences, embeddings1, range(3)):
    sim = 1 - distance.cosine(emb1, emb2)  # Cosine similarity is the complement of cosine distance
    print(f"Sentence: {sentence}")
    print(f"Cosine Similarity: {sim:.4f}")
    print("")

# print(model2.device)

# should be working perfectly :))