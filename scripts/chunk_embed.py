import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

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

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

if __name__ == "__main__":
    with open("textbook1.txt", "r") as file:
        text = file.read()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, model)
    print("Chunks:", chunks[:2])
    print("Embeddings:", embeddings[:2])
