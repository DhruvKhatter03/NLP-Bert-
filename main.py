from scripts.extract_text import extract_text_from_pdf
from scripts.chunk_embed import chunk_text, embed_chunks
from scripts.cluster_summarize import cluster_and_summarize
from scripts.milvus_operations import connect_milvus, create_collection, insert_data
from scripts.retrieval import hybrid_retrieval
from sentence_transformers import SentenceTransformer, CrossEncoder
from scripts.question_answering import answer_question

def main():
    textbooks = ["textbook1.pdf", "textbook2.pdf", "textbook3.pdf"]
    texts = [extract_text_from_pdf(tb) for tb in textbooks]
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    all_chunks, all_embeddings = [], []
    for idx, text in enumerate(texts):
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks, model)
        embeddings, clusters = cluster_and_summarize(embeddings, chunks, model)
        all_chunks.append(chunks)
        all_embeddings.append(embeddings)
    
    connect_milvus()
    collection = create_collection()
    
    for idx, embeddings in enumerate(all_embeddings):
        insert_data(collection, embeddings, idx + 1)
    
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    query = "Explain the concept of machine learning."
    results = hybrid_retrieval(query, all_chunks[0], all_embeddings[0], cross_encoder_model)
    context = " ".join(results[:5])
    answer = answer_question(query, context)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
