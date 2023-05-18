from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

model_path = "./llama_models_hf"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

# Preprocess the data
def preprocess(data):
    return tokenizer(data['text'], truncation=True, padding=False, max_length=512)

train_dataset = dataset['train'].map(preprocess, batched=True)
test_dataset = dataset['test'].map(preprocess, batched=True)

# Set the format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

