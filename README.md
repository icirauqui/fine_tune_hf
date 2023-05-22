# Fine Tune HF LLMs

Fine tune Large Language Models in HuggingFace format.
If you have models in different formats, you'll need to transform them to the HuggingFace's. 
You may do so with the script provided by HuggingFace:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
python3 src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

## Requirements

Provided in requirements.txt:

```bash
pip3 install -r requirements.txt
```


## Run

```bash
python3 main.py
```

## Usage

 - Button "Input Model": select the location of your weights, alternatively write the HF model name, e.g., "decapoda-research/llama-7b-hf".
 - Button "Output Path": select the path to save the fine-tuned model.
 - Button "Data path": select the path to the data.
 - CheckBox "GPU": use GPU or not.
 - Button "Train": begin training, after completion the ressulting model is saved in the Output Path.
 - Button "Evaluate": evaluate the model on the test set.
 - Button "Inference": use the model to generate text without further training.


## Notes

Model and LMHead are set to work on the same DEVICE; you can change this by setting the DEVICE in the device_map variable in finetuner.py to "cpu" or "cuda".

Alternatively, you can specify the device element-wise, e.g.:

```python
device_map = {
    'model.layers.0.self_attn.q_proj.weight': 'cpu',
    'model.layers.0.self_attn.v_proj.weight': 'cpu',
    'model.layers.0.self_attn.k_proj.weight': 'cpu',
    'model.layers.0.self_attn.o_proj.weight': 'cpu',
    'model.layers.0.mlp.gate_proj.weight': 'cpu',
    'model.layers.0.mlp.down_proj.weight': 'cpu',
    'model.layers.0.mlp.up_proj.weight': 'cpu',
    'model.layers.0.input_layernorm.weight': 'cpu',
    'model.layers.0.post_attention_layernorm.weight': 'cpu',
    'model.layers.0.self_attn.rotary_emb.inv_freq': 'cpu',
}
```

