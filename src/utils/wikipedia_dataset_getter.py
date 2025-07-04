"""
Provides a getter for the Hugging Face Wikipedia dataset.
By default, fetches the first 500 elements, but allows parameterization.
"""
from typing import List, Dict, Any

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install the 'datasets' package: pip install datasets")

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

def get_wikipedia_chromadb_with_embeddings(
    num_articles: int = 5000,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    collection_name: str = "wikipedia_rag_data"
):
    """
    Fetches articles, splits them into chunks, embeds them, and stores in a chromadb collection.
    Each chunk is stored with metadata indicating its article index.

    Args:
        num_articles (int): Number of articles to embed. Defaults to 5000.
        chunk_size (int): Size of each chunk. Defaults to 1000.
        chunk_overlap (int): Overlap between chunks. Defaults to 50.
        collection_name (str): Name for the chromadb collection. Defaults to "wikipedia_rag_data".

    Returns:
        chromadb.api.models.Collection.Collection: The chromadb collection with embedded chunks and metadata.
    """
    # Load articles
    dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    articles = dataset.select(range(num_articles))
    texts = [a['text'] for a in articles]
    titles = [a['title'] for a in articles]

    # Split into chunks and build metadata
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadatas = []
    for idx, doc in enumerate(texts):
        doc_chunks = splitter.split_text(doc)
        chunks.extend(doc_chunks)
        metadatas.extend([
            {"article_index": idx, "title": titles[idx]} for _ in doc_chunks
        ])

    # Embed chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    # Store in chromadb
    client = chromadb.Client()
    collection = client.create_collection(name=collection_name)
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=[f"chunk{i}" for i in range(len(chunks))]
    )
    return collection

def get_wikipedia_items(num_items: int = 500) -> List[Dict[str, Any]]:
    """
    Fetches the first `num_items` items from the Wikipedia dataset on Hugging Face.

    Args:
        num_items (int): Number of items to fetch. Defaults to 500.

    Returns:
        List[Dict[str, Any]]: List of Wikipedia dataset items.
    """
    dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    return dataset.select(range(num_items))
