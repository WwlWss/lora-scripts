# Anima training in SD-Trainer

This branch adds both **Anima LoRA** and **Anima full finetune** support to the existing SD-Trainer GUI while keeping the original SD / SDXL / SD3 / FLUX / Chroma paths intact.

## What changed

- Adds `kohya-ss/sd-scripts` as a pinned git submodule.
- Uses `sd-scripts/anima_train_network.py` for Anima LoRA.
- Uses `sd-scripts/anima_train.py` for Anima full finetune.
- Adds Anima model detection and validation.
- Extends the existing **Flux LoRA expert page** with a `model_type = anima` option. The packaged frontend is still the upstream prebuilt VuePress frontend, so the page title remains Flux; the form itself is provided dynamically by the backend schema and switches to Anima-specific fields.
- When Anima is selected, the GUI exposes a second selector for `LoRA` vs `full finetune`.
- Full finetune hides LoRA rank/alpha/network settings and exposes Anima component learning rates: self-attention, cross-attention, MLP, AdaLN modulation and LLM Adapter.
- Adds Qwen3-0.6B, Qwen-Image VAE, LLM Adapter/T5 tokenizer, timestep, attention, VAE, caching and block-swap controls.
- Adds both `Anima LoRA 训练` and `Anima 全参微调` presets.
- Updates Python dependencies to versions compatible with the current Anima implementation in `sd-scripts`.

## Install / update

After checking out this branch, initialize all submodules:

```bash
git submodule update --init --recursive
```

Then run the normal installer again so the upgraded Python dependencies are installed.

Windows:

```powershell
.\install.ps1
```

Linux:

```bash
bash install.bash
```

The GUI also attempts to initialize missing submodules automatically at startup unless environment preparation is skipped.

## Training Anima from the GUI

1. Open **LoRA training -> Flux** (expert page).
2. Set **model architecture** to `anima`.
3. Choose **Anima training mode**:
   - `lora`: routes to `sd-scripts/anima_train_network.py`.
   - `finetune`: routes to `sd-scripts/anima_train.py`.
4. Select the Anima DiT checkpoint in `pretrained_model_name_or_path`.
5. Select the Qwen3-0.6B text encoder in `qwen3`.
6. Select the Qwen-Image VAE in `vae`.
7. Optionally provide a separate LLM Adapter or T5 tokenizer directory.
8. Configure the dataset and output settings, or load one of the included Anima presets.
9. Start training.

### LoRA mode

LoRA mode uses `networks.lora_anima` and shows the usual LoRA controls such as rank, alpha, dropout and network arguments.

The included LoRA preset follows the current `sd-scripts` Anima LoRA example closely:

- rank: `8`
- alpha: `1`
- learning rate: `1e-4`
- optimizer: `AdamW8bit`
- scheduler: `constant`
- timestep sampling: `sigmoid`
- mixed precision: `bf16`
- gradient checkpointing: enabled
- latent cache: enabled
- text encoder output cache: enabled
- Qwen-Image 2D VAE: enabled
- VAE chunk size: `64`

### Full finetune mode

Full finetune directly trains the Anima DiT and does not use `network_module`, `network_dim` or `network_alpha`.

The official `anima_train.py` keeps the Qwen3 text encoder frozen. The GUI exposes the script's per-component learning rates:

- `self_attn_lr`
- `cross_attn_lr`
- `mlp_lr`
- `mod_lr`
- `llm_adapter_lr`

Leaving one of these blank makes that component use the base `learning_rate`; setting it to `0` freezes that component.

The included `Anima 全参微调` preset starts conservatively with a base learning rate of `1e-5`, BF16 full-precision model weights, gradient checkpointing and both latent/text-output caching enabled. Adjust the learning rate and memory controls for your dataset and GPU.

## Memory controls

For lower VRAM cards, `blocks_to_swap` can be increased. The 28-block Anima model supports at most 26 swapped blocks, and the 32-block model supports at most 30.

`blocks_to_swap`, `cpu_offload_checkpointing` and `unsloth_offload_checkpointing` are mutually constrained by upstream `sd-scripts`; do not enable block swap together with either checkpoint-offload mode.

## Updating sd-scripts later

The submodule is deliberately pinned to a known upstream commit so a future `sd-scripts` change cannot silently break the GUI. To update it, advance the `sd-scripts` gitlink deliberately and retest the schema/arguments before merging that change.
