# Fine Tune HF LLMs

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

