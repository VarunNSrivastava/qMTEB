import os
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial import distance
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 1 for offline


def load_model_and_tokenizer(filepath):
    model = AutoModel.from_pretrained(filepath)
    tokenizer = AutoTokenizer.from_pretrained(filepath)
    return model, tokenizer


def get_sentence_embeddings(sentences, model, tokenizer):
    tokens = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    return embeddings


def compare_embeddings(path1, path2):
    model, tokenizer1 = load_model_and_tokenizer(path1)
    quantized_model, tokenizer2 = load_model_and_tokenizer(path2)

    sentence_embeddings1 = get_sentence_embeddings(sentences, model, tokenizer1)
    sentence_embeddings2 = get_sentence_embeddings(sentences, quantized_model, tokenizer2)

    for sentence, emb1, emb2 in zip(sentences, sentence_embeddings1, sentence_embeddings2):
        cosine_similarity = 1 - distance.cosine(emb1, emb2)  # scipy's cosine returns dissimilarity
        euclidean_distance = distance.euclidean(emb1, emb2)

        print("Sentence:", sentence)
        print("Embedding1 shape:", emb1.shape)
        print("Embedding2 shape:", emb2.shape)
        print("Cosine Similarity:", cosine_similarity)
        print("Euclidean Distance:", euclidean_distance)
        print("--------")


# Testing the comparison function
model_filepath = "./models/all-MiniLM-L6-v2"
quantized_model_filepath = "./models/all-MiniLM-L6-v2-q8"
sentences = [
    'This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.'
]

compare_embeddings(model_filepath, quantized_model_filepath)