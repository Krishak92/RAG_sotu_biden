import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def encode_query(query, encoder_model):
    """
    Encode a question into a vector with SentenceTransformer 

    Args:
        query (str): Question asked.
        encoder_model (SentenceTransformer): Semantic encoding model

    Returns:
        np.ndarray: encoded vector of the question
    """
    return encoder_model.encode([query], convert_to_numpy=True)[0]

def retrieve_passages(query_vector, faiss_index, passages, top_k=5):
    """
    Search the most pertinents passages in the FAISS index

    Args:
        query_vector (np.ndarray): Encoded vector of the question
        faiss_index (faiss.Index): FAISS index containing the passages vectors
        passages (list): List of passages corresponding to the vectors
        top_k (int): Number of passages to recover

    Returns:
        list: List of most pertinents passages
    """
    _, indices = faiss_index.search(query_vector.reshape(1, -1), top_k)
    return [passages[i] for i in indices[0]]

def generate_response(context, question, model, tokenizer, max_length=200):
    """
    Generate an answer using a generative model and a context

    Args:
        context (str): Context based on the recuperate passages
        question (str): Question asked
        model: Generative model pre-trained or fine-tuned
        tokenizer: Tokenizer corresponding to the generative model
        max_length (int): Maximal length of the generated answer

    Returns:
        str: Answer generated
    """
    input_text = f"Question: {question}\nContext: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def pipeline(question, faiss_index_path, passages_path, encoder_model_name="all-MiniLM-L6-v2", gen_model_name="t5-small"):
    """
    Implement the RAG pipeline to answer to a question

    Args:
        question (str): Question asked by the user
        faiss_index_path (str): Path to the FAISS index
        passages_path (str): Path to the file containing the passages
        encoder_model_name (str): Name of the model SentenceTransformer for the enconding
        gen_model_name (str): Name of the generative model Hugging Face

    Returns:
        str: Answer generated by the pipeline RAG
    """
    #load FAISS index and passages
    faiss_index = faiss.read_index(faiss_index_path)
    passages = np.load(passages_path, allow_pickle=True)

    #load encoding and generation models
    encoder_model = SentenceTransformer(encoder_model_name)
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

    #step 1: search
    query_vector = encode_query(question, encoder_model)
    relevant_passages = retrieve_passages(query_vector, faiss_index, passages, top_k=5)
    context = " ".join(relevant_passages)

    #step 2: answer
    response = generate_response(context, question, model, tokenizer)
    return response