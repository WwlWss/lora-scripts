Schema.intersect([
    Schema.object({
        model_type: Schema.union(["flux", "chroma", "anima"]).default("flux").description("模型架构：FLUX / Chroma / Anima"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/model.safetensors").description("底模路径；Anima 请选择 DiT safetensors"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("从某个 save_state 保存的中断状态继续训练"),
    }).description("训练用模型"),

    Schema.union([
        Schema.object({
            model_type: Schema.union(["flux", "chroma"]).required(),
            ae: Schema.string().role('filepicker', { type: "model-file" }).description("AE 模型文件路径"),
            clip_l: Schema.string().role('filepicker', { type: "model-file" }).description("CLIP-L 模型文件路径"),
            t5xxl: Schema.string().role('filepicker', { type: "model-file" }).description("T5-XXL 模型文件路径"),
        }),
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.union(["lora", "finetune"]).default("lora").description("Anima 训练方式：LoRA 或全参微调（全参直接训练 DiT，Qwen3 文本编码器按官方脚本保持冻结）"),
            qwen3: Schema.string().role('filepicker', { type: "model-file" }).required().description("Anima 文本编码器：Qwen3-0.6B safetensors 或本地 HuggingFace 模型目录"),
            vae: Schema.string().role('filepicker', { type: "model-file" }).required().description("Anima VAE：Qwen-Image VAE safetensors / pth"),
            llm_adapter_path: Schema.string().role('filepicker', { type: "model-file" }).description("可选：独立 LLM Adapter 权重；留空时从 DiT 中读取"),
            t5_tokenizer_path: Schema.string().role('filepicker', { type: "folder" }).description("可选：T5 tokenizer 目录；留空使用 sd-scripts 内置配置"),
        }),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.union(["flux", "chroma"]).required(),
            model_train_type: Schema.string().default("flux-lora").disabled().description("实际训练种类"),
        }),
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("lora").required(),
            model_train_type: Schema.string().default("anima-lora").disabled().description("实际训练种类"),
        }),
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            model_train_type: Schema.string().default("anima-finetune").disabled().description("实际训练种类"),
        }),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.union(["flux", "chroma"]).required(),
            timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift"]).default("sigmoid").description("时间步采样"),
            sigmoid_scale: Schema.number().step(0.001).default(1.0).description("sigmoid 缩放"),
            model_prediction_type: Schema.union(["raw", "additive", "sigma_scaled"]).default("raw").description("模型预测类型"),
            discrete_flow_shift: Schema.number().step(0.001).default(1.0).description("Euler 调度器离散流位移"),
            loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("损失函数类型"),
            guidance_scale: Schema.number().step(0.01).default(1.0).description("CFG 引导缩放"),
            t5xxl_max_token_length: Schema.number().step(1).description("T5XXL 最大 token 长度（不填写使用自动）"),
            train_t5xxl: Schema.boolean().default(false).description("训练 T5XXL（不推荐）"),
            apply_t5_attn_mask: Schema.boolean().default(true).description("对 T5-XXL 编码器和 FLUX double block 应用注意力掩码"),
        }).description("Flux / Chroma 专用参数"),
        Schema.object({
            model_type: Schema.const("anima").required(),
            timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift", "flux_shift"]).default("sigmoid").description("Anima Rectified Flow 时间步采样"),
            sigmoid_scale: Schema.number().step(0.001).default(1.0).description("sigmoid / shift / flux_shift 缩放"),
            discrete_flow_shift: Schema.number().step(0.001).default(1.0).description("Rectified Flow 离散流位移"),
            qwen3_max_token_length: Schema.number().min(1).step(1).default(512).description("Qwen3 最大 token 长度"),
            t5_max_token_length: Schema.number().min(1).step(1).default(512).description("T5 tokenizer 最大 token 长度"),
            attn_mode: Schema.union(["torch", "xformers", "flash"]).default("torch").description("Attention 实现；xformers 需要同时启用 split_attn"),
            split_attn: Schema.boolean().default(false).description("拆分 attention 计算以降低显存；xformers 模式必须开启"),
            vae_chunk_size: Schema.number().min(2).step(2).default(64).description("Qwen-Image VAE 空间分块大小；越小越省显存"),
            vae_disable_cache: Schema.boolean().default(true).description("关闭 Qwen-Image VAE 内部缓存以减少显存"),
            qwen_image_vae_2d: Schema.boolean().default(true).description("使用图像专用 2D Qwen-Image VAE；单图训练推荐，速度更快且显存更低"),
            blocks_to_swap: Schema.number().min(0).max(30).step(1).default(0).description("将 DiT block 交换到 CPU；0 为关闭。28-block 模型最多 26，32-block 模型最多 30"),
            unsloth_offload_checkpointing: Schema.boolean().default(false).description("将 checkpoint activation 异步卸载到 CPU；不能和 blocks_to_swap 同时使用"),
            cuda_allow_tf32: Schema.boolean().default(true).description("Ampere 及更新显卡允许 TF32"),
        }).description("Anima 专用参数"),
    ]),

    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            resolution: Schema.string().default("768,768").description("训练图片分辨率，宽x高。Anima 建议从 1024,1024 起按数据集调整。"),
            enable_bucket: Schema.boolean().default(true).description("启用 arb 桶以允许非固定宽高比的图片"),
            min_bucket_reso: Schema.number().default(256).description("arb 桶最小分辨率"),
            max_bucket_reso: Schema.number().default(2048).description("arb 桶最大分辨率"),
            bucket_reso_steps: Schema.number().default(64).description("arb 桶分辨率划分单位"),
        })
    ).description("数据集设置"),

    SHARED_SCHEMAS.SAVE_SETTINGS,

    Schema.object({
        max_train_epochs: Schema.number().min(1).default(20).description("最大训练 epoch（轮数）"),
        train_batch_size: Schema.number().min(1).default(1).description("批量大小, 越高显存占用越高"),
        gradient_checkpointing: Schema.boolean().default(true).description("梯度检查点"),
        gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累加步数"),
    }).description("训练相关参数"),

    SHARED_SCHEMAS.LR_OPTIMIZER,

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            self_attn_lr: Schema.string().description("Self-Attention 学习率；留空=总学习率，0=冻结该组件"),
            cross_attn_lr: Schema.string().description("Cross-Attention 学习率；留空=总学习率，0=冻结该组件"),
            mlp_lr: Schema.string().description("MLP 学习率；留空=总学习率，0=冻结该组件"),
            mod_lr: Schema.string().description("AdaLN modulation 学习率；留空=总学习率，0=冻结该组件"),
            llm_adapter_lr: Schema.string().description("LLM Adapter 学习率；留空=总学习率，0=冻结 Adapter"),
            cpu_offload_checkpointing: Schema.boolean().default(false).description("将 gradient-checkpoint activation 卸载到 CPU；不能与 blocks_to_swap / unsloth_offload_checkpointing 同时使用"),
        }).description("Anima 全参微调分组件学习率"),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.intersect([
            Schema.object({
                model_type: Schema.union(["flux", "chroma"]).required(),
                network_module: Schema.union(["networks.lora_flux", "networks.oft_flux", "lycoris.kohya"]).default("networks.lora_flux").description("训练网络模块"),
                network_weights: Schema.string().role('filepicker').description("从已有的 LoRA 模型上继续训练，填写路径"),
                network_dim: Schema.number().min(1).default(2).description("网络维度，常用 4~128"),
                network_alpha: Schema.number().min(1).default(16).description("LoRA alpha"),
                network_dropout: Schema.number().step(0.01).default(0).description('dropout 概率'),
                scale_weight_norms: Schema.number().step(0.01).min(0).description("最大范数正则化。如果使用，推荐为 1"),
                network_args_custom: Schema.array(String).role('table').description('自定义 network_args，一行一个'),
                enable_base_weight: Schema.boolean().default(false).description('启用基础权重（差异炼丹）'),
            }).description("网络设置"),
            SHARED_SCHEMAS.LYCORIS_MAIN,
            SHARED_SCHEMAS.LYCORIS_LOKR,
            SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("lora").required(),
                network_module: Schema.string().default("networks.lora_anima").disabled().description("Anima LoRA 网络模块"),
                network_weights: Schema.string().role('filepicker').description("从已有的 Anima LoRA 上继续训练"),
                network_dim: Schema.number().min(1).default(8).description("LoRA rank；官方示例为 8，可按数据规模提高"),
                network_alpha: Schema.number().min(1).default(1).description("LoRA alpha；新版 sd-scripts 官方示例为 1"),
                network_dropout: Schema.number().step(0.01).default(0).description("LoRA dropout"),
                scale_weight_norms: Schema.number().step(0.01).min(0).description("最大范数正则化"),
                network_args_custom: Schema.array(String).role('table').description("Anima network_args；可填写 train_llm_adapter=True、network_reg_dims=...、network_reg_lrs=... 等，一行一个"),
                enable_base_weight: Schema.boolean().default(false).description('启用基础权重（差异炼丹）'),
                network_train_unet_only: Schema.boolean().default(true).description("仅训练 Anima DiT 的 LoRA"),
                network_train_text_encoder_only: Schema.boolean().default(false).description("仅训练文本编码器网络；Anima 通常保持关闭"),
            }).description("Anima LoRA 网络设置"),
            SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
        ]),
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
        }).description("Anima 全参微调直接训练 DiT，不使用 LoRA network 参数"),
    ]),

    SHARED_SCHEMAS.PREVIEW_IMAGE,
    SHARED_SCHEMAS.LOG_SETTINGS,

    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {}, ["max_token_length"])).description("caption（Tag）选项"),

    SHARED_SCHEMAS.NOISE_SETTINGS,
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,
    SHARED_SCHEMAS.OTHER,

    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            Schema.object(
                UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {
                    fp8_base: Schema.boolean().default(true).description("对基础模型使用 FP8 精度"),
                    fp8_base_unet: Schema.boolean().description("仅对 U-Net 使用 FP8 精度（CLIP-L不使用）"),
                    sdpa: Schema.boolean().default(true).description("启用 sdpa"),
                    cache_text_encoder_outputs: Schema.boolean().default(true).description("缓存文本编码器输出；使用时需要关闭 shuffle_caption"),
                    cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("缓存文本编码器输出到磁盘"),
                }, ["xformers"])
            ).description("Flux / Chroma 速度优化选项"),
        ]),
        Schema.intersect([
            Schema.object({ model_type: Schema.const("anima").required() }),
            Schema.object(
                UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {
                    cache_latents: Schema.boolean().default(true).description("缓存 Qwen-Image VAE latent"),
                    cache_latents_to_disk: Schema.boolean().default(true).description("将 latent 缓存到磁盘"),
                    cache_text_encoder_outputs: Schema.boolean().default(true).description("缓存 Qwen3 输出以释放显存；启用后需关闭 shuffle_caption"),
                    cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("将 Qwen3 输出缓存到磁盘"),
                    vae_batch_size: Schema.number().min(1).default(1).description("VAE 编码批量大小"),
                }, ["xformers", "sdpa"])
            ).description("Anima 速度与缓存选项"),
        ]),
    ]),

    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);