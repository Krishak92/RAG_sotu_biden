import time
from src.pipeline import encode_query, retrieve_passages, generate_response

def test_pipeline(faiss_index, passages, encoder_model, generative_model, tokenizer, query, top_k=5):
    """
    Évalue la pipeline de question-réponse avec récupération et génération.

    Args:
        faiss_index (faiss.Index): Index FAISS pour la récupération des passages.
        passages (list): Liste des passages correspondant aux vecteurs dans l'index.
        encoder_model (SentenceTransformer): Modèle d'encodage des questions.
        generative_model: Modèle génératif (pré-entraîné ou fine-tuné).
        tokenizer: Tokenizer associé au modèle génératif.
        query (str): Question posée.
        top_k (int): Nombre de passages les plus pertinents à récupérer.

    Returns:
        None
    """
    print(f"Query: {query}\n")

    # Mesure du temps de récupération
    time_1 = time.time()
    query_vector = encode_query(query, encoder_model)
    retrieved_passages = retrieve_passages(query_vector, faiss_index, passages, top_k)
    time_2 = time.time()

    # Combiner les passages récupérés pour former le contexte
    context = " ".join(retrieved_passages)

    # Mesure du temps de génération
    time_3 = time.time()
    response = generate_response(context, query, generative_model, tokenizer)
    time_4 = time.time()

    # Afficher les résultats
    print(f"Retrieved Passages:\n{retrieved_passages}\n")
    print(f"Generated Response:\n{response}\n")
    print(f"Retrieval Time: {round(time_2 - time_1, 3)} sec.")
    print(f"Generation Time: {round(time_4 - time_3, 3)} sec.")
    print(f"Total Inference Time: {round(time_4 - time_1, 3)} sec.")
