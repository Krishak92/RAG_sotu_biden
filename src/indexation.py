import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def split_into_passages(text, passage_size=100):
    """
    Split a text into passages of fixed size

    Args:
        text (str): Raw text
        passage_size (int): Size of a passage (number of words)

    Returns:
        list: List of passages
    """
    words = text.split()
    passages = [
        ' '.join(words[i:i + passage_size]) for i in range(0, len(words), passage_size)
    ]
    return passages

def encode_passages(passages, model_name="all-MiniLM-L6-v2"):
    """
    Encode passages as embeddings using SentenceTransformers

    Args:
        passages (list): List of passages
        model_name (str): Name of the SentenceTransformers model

    Returns:
        np.ndarray: Matrix of encoded vectors
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(passages, show_progress_bar=True)
    return np.array(embeddings)

def build_faiss_index(embeddings):
    """
    Create a FAISS index from encoded vectors

    Args:
        embeddings (np.ndarray): Matrix of encoded vectors

    Returns:
        faiss.IndexFlatL2: FAISS index
    """
    d = embeddings.shape[1]  #Vectors shape
    index = faiss.IndexFlatL2(d)  #index with euclidean distance
    index.add(embeddings)  #add vectors to index
    return index