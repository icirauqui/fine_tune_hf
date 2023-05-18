import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


def main():
    print("Hello Trainer")

    model_name = "7B"
    model_path = "./llama_models_hf"

    print("Load tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print("Done")
    

    dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

    # Preprocess the data
    def preprocess(data):
        return tokenizer(data['text'], truncation=True, padding=False, max_length=512)

    train_dataset = dataset['train'].map(preprocess, batched=True)
    test_dataset = dataset['test'].map(preprocess, batched=True)

    # Set the format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        do_train=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        weight_decay=0.01
        #half_precision_backend="auto",
    )

    # Create the trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    main()