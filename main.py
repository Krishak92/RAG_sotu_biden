import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from src.preprocessing import normalize_text
from src.indexation import split_into_passages, encode_passages, build_faiss_index
from src.training import train_model

def load_text(file_path):
    """
    Load raw text from a file

    Args:
        file_path (str): Path of the file.

    Returns:
        str: Content of the file as chains of character.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def main():
    
    official_transcript_path = "data/biden-sotu-2023-planned-official.txt"
    autogenerated_transcript_path = "data/biden-sotu-2023-autogenerated-transcript.txt"

    #loading texts
    official_text = load_text(official_transcript_path)
    autogenerated_text = load_text(autogenerated_transcript_path)
    # combined_text = official_text + "\n" + autogenerated_text

    #preprocessing
    normalized_official_transcript = normalize_text(official_text)
    normalized_autogenerated_transcript = normalize_text(autogenerated_text)

    with open("normalized_official_transcript.txt", 'w', encoding='utf-8') as file:
        file.write(normalized_official_transcript)

    with open("normalized_autogenerated_transcript.txt", 'w', encoding='utf-8') as file:
        file.write(normalized_autogenerated_transcript)

    print("Normalization over. Normalized files have been saved.")

    #split
    passages_official = split_into_passages(normalized_official_transcript, passage_size=100)
    passages_autogenerated = split_into_passages(normalized_autogenerated_transcript, passage_size=100)

    #encode
    embeddings_official = encode_passages(passages_official)
    embeddings_autogenerated = encode_passages(passages_autogenerated)

    #index construction
    index_official = build_faiss_index(embeddings_official)
    index_autogenerated = build_faiss_index(embeddings_autogenerated)

    #save
    np.save("passages_official.npy", passages_official)
    np.save("passages_autogenerated.npy", passages_autogenerated)
    faiss.write_index(index_official, "faiss_index_official.bin")
    faiss.write_index(index_autogenerated, "faiss_index_autogenerated.bin")

    print("FAISS index created and successfully saved!")

    #training
    trained_model = train_model(embeddings_official)

if __name__ == "__main__":
    main()