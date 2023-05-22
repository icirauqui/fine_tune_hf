import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModel
import os
import sys
from typing import List

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
 
import fire
#import torch
from datasets import load_dataset
import pandas as pd
 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#from pylab import rcParams


sns.set(rc={'figure.figsize':(10, 7)})
sns.set(rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)
    



def train(gpu = False, path_in = "", path_out = "", data_file = ""):

    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    import torch

    DEVICE = "cuda" if (torch.cuda.is_available() and gpu) else "cpu"
    print("Torch device:", DEVICE)

    #  - - - - - - - - WEIGHTS - - - - - - - - - - - -

    BASE_MODEL = "decapoda-research/llama-7b-hf"
    BASE_MODEL = "/home/icirauqui/wErkspace/llm/llama_models/hf/13B"
    OUTPUT_DIR = "experiments"
    DATA_PATH = "data/alpaca-bitcoin-sentiment-dataset.json"


    if path_in != "":
        BASE_MODEL = path_in
    if path_out != "":
        OUTPUT_DIR = path_out
    if data_file != "":
        DATA_PATH = data_file

    
    device_map = {
        'model': DEVICE,
        'lm_head': DEVICE,
    }

    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        #load_in_8bit_fp32_cpu_offload=True,
        llm_int8_enable_fp32_cpu_offload=True,
        torch_dtype=torch.float16,
        #device_map="auto",
        #device_map = ["auto", "balanced", "balanced_low_0", "sequential"]
        device_map=device_map,
    )

    print(model.get_memory_footprint())
    
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"


    #  - - - - - - - - DATASET - - - - - - - - - - - -

    data = load_dataset("json", data_files=DATA_PATH)
    print(data["train"])

    def generate_prompt(data_point):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
    ### Instruction:
    {data_point["instruction"]}
    ### Input:
    {data_point["input"]}
    ### Response:
    {data_point["output"]}"""
    
    CUTOFF_LEN = 512
    #CUTOFF_LEN = [512, 1024, 2048]
    
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
    
        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    train_val = data["train"].train_test_split(
        test_size=200, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].map(generate_and_tokenize_prompt)
    )




    #  - - - - - - - - TRAINING - - - - - - - - - - - -

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT= 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]
    
    BATCH_SIZE = 4 #128
    MICRO_BATCH_SIZE = 1 #4
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    LEARNING_RATE = 3e-4
    TRAIN_STEPS = 10 #300

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        #fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard"
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    
    model = torch.compile(model)

    print("START TRAINING")

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)




if __name__ == "__main__":
    train()
