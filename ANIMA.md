# Anima LoRA training in SD-Trainer

This branch adds Anima LoRA training support to the existing SD-Trainer GUI while keeping the original SD / SDXL / SD3 / FLUX / Chroma paths intact.

## What changed

- Adds `kohya-ss/sd-scripts` as a pinned git submodule and uses its `anima_train_network.py` for Anima jobs.
- Adds Anima model detection and validation.
- Extends the existing **Flux LoRA expert page** with a `model_type = anima` option. The packaged frontend is still the upstream prebuilt VuePress frontend, so the page title remains Flux; the form itself is provided dynamically by the backend schema and switches to Anima-specific fields.
- Adds Qwen3-0.6B, Qwen-Image VAE, LLM Adapter/T5 tokenizer, timestep, attention, VAE and block-swap controls.
- Adds an `Anima LoRA 训练` preset.
- Updates Python dependencies to versions compatible with the current Anima implementation in `sd-scripts`.

## Install / update

After checking out this branch, initialize all submodules:

```bash
git submodule update --init --recursive
```

Then run the normal installer again so the upgraded Python dependencies are installed:

Windows:

```powershell
.\install.ps1
```

Linux:

```bash
bash install.bash
```

The GUI also attempts to initialize missing submodules automatically at startup unless environment preparation is skipped.

## Training Anima LoRA from the GUI

1. Open **LoRA training -> Flux** (expert page).
2. Set **model architecture** to `anima`.
3. Select the Anima DiT checkpoint in `pretrained_model_name_or_path`.
4. Select the Qwen3-0.6B text encoder in `qwen3`.
5. Select the Qwen-Image VAE in `vae`.
6. Optionally provide a separate LLM Adapter or T5 tokenizer directory. Leaving these blank is supported.
7. Configure the dataset and output settings as usual, or load the `Anima LoRA 训练` preset.
8. Start training. The backend routes the job to `sd-scripts/anima_train_network.py` and uses `networks.lora_anima`.

## Default Anima settings

The included preset follows the current `sd-scripts` Anima LoRA example closely:

- rank: `8`
- alpha: `1`
- learning rate: `1e-4`
- optimizer: `AdamW8bit`
- scheduler: `constant`
- timestep sampling: `sigmoid`
- discrete flow shift: `1.0`
- mixed precision: `bf16`
- gradient checkpointing: enabled
- latent cache: enabled
- text encoder output cache: enabled
- Qwen-Image 2D VAE: enabled
- VAE chunk size: `64`

For lower VRAM cards, `blocks_to_swap` can be increased. The 28-block Anima model supports at most 26 swapped blocks. `blocks_to_swap` and `unsloth_offload_checkpointing` must not be enabled together.

## Updating sd-scripts later

The submodule is deliberately pinned to a known upstream commit so a future `sd-scripts` change cannot silently break the GUI. To update it, advance the `sd-scripts` gitlink deliberately and retest the schema/arguments before merging that change.
