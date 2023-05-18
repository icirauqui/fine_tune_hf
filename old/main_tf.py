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

    import tensorflow as tf
    from tensorflow import keras
    from keras.optimizers import Adam

    from transformers import TFAutoModelForCausalLM, AutoConfig

    model = AutoConfig.from_pretrained(model_path)
    model.compile(optimizer=Adam(3e-5))

    return

if __name__ == "__main__":
    main()