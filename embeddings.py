import os
from langchain_community.embeddings import OllamaEmbeddings
from config import EMBED_MODEL

def get_embedding_model():
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=base_url)

def embed_texts(texts: list) -> list:
    model = get_embedding_model()
    embeddings = model.embed_documents(texts)
    return embeddings

def embed_query(query: str) -> list:
    model = get_embedding_model()
    return model.embed_query(query)