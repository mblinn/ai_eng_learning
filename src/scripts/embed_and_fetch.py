"""
Interactive script to embed Wikipedia articles, store them in chromadb, and query for relevant chunks.
"""
import sys
from utils.wikipedia_dataset_getter import get_wikipedia_chromadb_with_embeddings
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    print("[INFO] Building chromadb and chunk map from Wikipedia articles...")
    collection = get_wikipedia_chromadb_with_embeddings(num_articles=100, chunk_size=500, chunk_overlap=50)
    print("[INFO] Ready for queries! Type your question and press Enter (Ctrl+C to exit)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    while True:
        try:
            user_query = input("\nYour query: ")
            query_embedding = model.encode([user_query])
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=5,
                include=["documents", "distances", "metadatas"]
            )
            print("\nTop 5 Chunks:")
            for idx, (doc, dist, meta) in enumerate(zip(results["documents"][0], results["distances"][0], results["metadatas"][0])):
                article_idx = meta.get("article_index", "?")
                title = meta.get("title", "?")
                print(f"[{idx+1}] article_index: {article_idx}, title: {title}, distance: {dist:.4f}\n{doc}\n{'-'*40}")
        except KeyboardInterrupt:
            print("\n[INFO] Exiting. Goodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            continue
