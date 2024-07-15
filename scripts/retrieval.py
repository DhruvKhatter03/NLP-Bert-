import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def chunk_text(text, max_tokens=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        if current_length + len(tokens) <= max_tokens:
            current_chunk.append(sentence)
            current_length += len(tokens)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(tokens)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def embed_chunks(chunks, model):
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def bm25_retrieval(query, chunks):
    tokenized_chunks = [nltk.word_tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = nltk.word_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_chunks = [chunks[idx] for idx in np.argsort(scores)[::-1]]
    return ranked_chunks

def hybrid_retrieval(query, chunks, embeddings, cross_encoder_model):
    bm25_results = bm25_retrieval(query, chunks)
    encoded_query = model.encode([query])
    scores = np.dot(embeddings, encoded_query.T).flatten()
    combined_scores = [(bm25_result, scores[idx]) for idx, bm25_result in enumerate(bm25_results)]
    reranked_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    reranked_chunks = [result[0] for result in reranked_results]
    return reranked_chunks

# Initialize models
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Example usage (for testing purposes)
if __name__ == "__main__":
    with open("textbook1.txt", "r") as file:
        text = file.read()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, model)
    query = "Explain the concept of machine learning."
    results = hybrid_retrieval(query, chunks, embeddings, cross_encoder_model)
    print("Top results:", results[:5])
