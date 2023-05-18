import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


def main():
    model_path = "./llama_models_hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)



    dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
    print("\n" + "*"*50 + "\n" + "Dataset")
    print(dataset)
    print("\n" + "*"*50 + "\n")


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("\n" + "*"*50 + "\n" + "Tokenized Dataset")
    print(tokenized_datasets)
    print("\n" + "*"*50 + "\n")

    dataset_train = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    dataset_test = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

    from transformers import DefaultDataCollator
    data_collator = DefaultDataCollator(return_tensors="tf")

    tf_train_dataset = dataset_train.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["labels"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )

    tf_validation_dataset = dataset_test.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["labels"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
    )

    import tensorflow as tf
    from transformers import TFAutoModelForCausalLM, TFAutoModel, GPT2Config

    config = GPT2Config.from_pretrained(model_path)
    model = TFAutoModelForCausalLM.from_pretrained(model_path, config=config)

    return

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)

    return





    import tensorflow as tf
    from tensorflow import keras
    from keras.optimizers import Adam

    from transformers import TFAutoModelForCausalLM

    model = TFAutoModelForCausalLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.compile(optimizer=Adam(3e-5))

    return

if __name__ == "__main__":
    main()