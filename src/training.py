import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset

def preprocess_data(examples, tokenizer, max_input_length=512, max_target_length=128):
    """
    Prétraite les données pour l'entraînement.

    Args:
        examples (dict): Exemple brut.
        tokenizer: Tokenizer Hugging Face.
        max_input_length (int): Longueur maximale des entrées.
        max_target_length (int): Longueur maximale des sorties.

    Returns:
        dict: Données tokenisées.
    """
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Tokeniser les cibles avec le même tokenizer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(dataset, output_dir="fine_tuned_model"):
    """
    Fine-tune le modèle sur les données fournies.

    Args:
        dataset: Ensemble de données prétraitées.
        tokenizer: Tokenizer Hugging Face.
        model: Modèle Hugging Face.
        output_dir (str): Répertoire où sauvegarder le modèle.

    Returns:
        None
    """
    #Split data into training and validation sets
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    #data preprocessing

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    tokenized_train = train_dataset.map(
        lambda x: preprocess_data(x, tokenizer), batched=True
    )
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_data(x, tokenizer), batched=True
    )

    #training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        push_to_hub=False,
    )

    #training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved in: {output_dir}")