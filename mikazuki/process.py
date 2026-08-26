
import asyncio
import os
import sys
from typing import Optional

import toml

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm


ANIMA_LORA_ONLY_KEYS = {
    "network_module",
    "network_weights",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "scale_weight_norms",
    "network_args",
    "network_args_custom",
    "network_train_unet_only",
    "network_train_text_encoder_only",
    "enable_base_weight",
    "base_weights",
    "base_weights_multiplier",
    "unet_lr",
    "text_encoder_lr",
}

ANIMA_OPTIONAL_FINETUNE_LRS = {
    "self_attn_lr",
    "cross_attn_lr",
    "mlp_lr",
    "mod_lr",
    "llm_adapter_lr",
}


def _resolve_anima_trainer(toml_path: str, trainer_file: str) -> str:
    """Switch Anima jobs between LoRA and full finetune before launching sd-scripts.

    The packaged frontend only has the existing Flux expert route, so the GUI sends a
    lightweight `anima_training_mode` routing field. This field is removed from the
    config before sd-scripts sees it. Full finetune also strips every LoRA-only option
    that may have been left in a loaded preset or browser form state.
    """
    if not trainer_file.replace("\\", "/").endswith("/sd-scripts/anima_train_network.py"):
        return trainer_file

    try:
        config = toml.load(toml_path)
    except Exception as e:
        log.warning(f"Unable to inspect Anima training mode, falling back to LoRA: {e}")
        return trainer_file

    mode = str(config.pop("anima_training_mode", "lora")).lower()
    if mode not in {"lora", "finetune"}:
        log.warning(f"Unknown Anima training mode '{mode}', falling back to LoRA")
        mode = "lora"

    if mode == "finetune":
        for key in ANIMA_LORA_ONLY_KEYS:
            config.pop(key, None)

        # Empty optional component-LR fields must be omitted, otherwise argparse's
        # float conversion sees an empty string instead of the intended None value.
        for key in ANIMA_OPTIONAL_FINETUNE_LRS:
            if config.get(key) in (None, ""):
                config.pop(key, None)

        trainer_file = "./sd-scripts/anima_train.py"
        if not os.path.exists(trainer_file):
            raise FileNotFoundError(
                "Anima full finetune script is missing. Run `git submodule update --init --recursive`."
            )
        log.info("Anima full finetune selected; using sd-scripts/anima_train.py")
    else:
        # Component LRs belong to full finetune only. Remove stale values when a
        # finetune preset was loaded and the user switched the form back to LoRA.
        for key in ANIMA_OPTIONAL_FINETUNE_LRS | {"cpu_offload_checkpointing"}:
            config.pop(key, None)
        log.info("Anima LoRA selected; using sd-scripts/anima_train_network.py")

    # `anima_training_mode` is a GUI-only routing key and must never reach sd-scripts.
    with open(toml_path, "w", encoding="utf-8") as f:
        toml.dump(config, f)

    return trainer_file


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")

    try:
        trainer_file = _resolve_anima_trainer(toml_path, trainer_file)
    except Exception as e:
        log.error(f"Failed to prepare Anima training / Anima 训练准备失败: {e}")
        return APIResponse(status="error", message=str(e))

    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"
    customize_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if gpu_ids:
        customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")

        if len(gpu_ids) > 1:
            args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_ids))]
            if sys.platform == "win32":
                customize_env["USE_LIBUV"] = "0"
                args[3:3] = ["--rdzv_backend", "c10d"]

    if not (task := tm.create_task(args, customize_env)):
        return APIResponse(status="error", message="Failed to create task / 无法创建训练任务")

    def _run():
        try:
            task.execute()
            result = task.communicate()
            if result.returncode != 0:
                log.error(f"Training failed / 训练失败")
            else:
                log.info(f"Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(status="success", message=f"Training started / 训练开始 ID: {task.task_id}")
