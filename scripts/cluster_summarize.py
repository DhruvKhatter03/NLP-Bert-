from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
import openai
import nltk
from nltk.tokenize import sent_tokenize

openai.api_key = 'sk-proj-6gem5qtG1ytYnnmrHbJoT3BlbkFJBo0vJfcXvgiD8Z3Q0nmA'

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

def summarize_text(text, engine="text-davinci-003"):
    response = openai.Completion.create(
        engine=engine,
        prompt=f"Summarize the following text:\n{text}",
        max_tokens=150
    )
    summary = response.choices[0].text.strip()
    return summary

def cluster_and_summarize(embeddings, chunks, model, n_components=10, n_iters=3):
    for _ in range(n_iters):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(embeddings)
        cluster_labels = gmm.predict(embeddings)
        
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[idx])
        
        summarized_chunks = []
        for cluster, texts in clusters.items():
            summary = summarize_text(" ".join(texts))
            summarized_chunks.append(summary)
        
        embeddings = model.encode(summarized_chunks, convert_to_tensor=True)
        chunks = summarized_chunks

    return embeddings, clusters

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

if __name__ == "__main__":
    with open("textbook1.txt", "r") as file:
        text = file.read()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, model)
    embeddings, clusters = cluster_and_summarize(embeddings, chunks, model)
    print("Clusters:", clusters)
