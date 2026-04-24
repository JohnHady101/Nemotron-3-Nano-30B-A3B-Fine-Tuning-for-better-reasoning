class Config:
    # Model
    BASE_MODEL   = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")   # swap to any causal LM
    DATASET_PATH = "/kaggle/input/datasets/johnhady101/glm5-0-reasoning/train-00000-of-00001.parquet"
    OUTPUT_DIR   = "./lora-finetuned"
    MERGED_DIR   = "./merged-model"

    # Precision — pick ONE:
    #   torch.float32  → safest, highest VRAM usage
    #   torch.float16  → faster on older GPUs (Volta / Turing)
    #   torch.bfloat16 → preferred on Ampere+ (A100, RTX 3090+)
    TORCH_DTYPE  = torch.bfloat16

    # Tokenisation
    MAX_LENGTH   = 2048
    TRUNCATION   = True

    # LoRA
    LORA_R          = 16        # rank
    LORA_ALPHA      = 32        # scaling factor (usually 2× rank)
    LORA_DROPOUT    = 0.05
    TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]   # LLaMA-style

    # Training
    NUM_EPOCHS     = 3
    BATCH_SIZE     = 2
    GRAD_ACCUM     = 8          # effective batch = BATCH_SIZE × GRAD_ACCUM
    LEARNING_RATE  = 2e-4
    WARMUP_RATIO   = 0.03
    LR_SCHEDULER   = "cosine"
    WEIGHT_DECAY   = 0.001
    MAX_GRAD_NORM  = 0.3
    LOGGING_STEPS  = 10
    SAVE_STEPS     = 100
    EVAL_STEPS     = 100
    # fp16/bf16 flags passed to TrainingArguments
    FP16           = False      # set True if TORCH_DTYPE is float16
    BF16           = True       # set True if TORCH_DTYPE is bfloat16
    VAL_SPLIT      = 0.1
    SEED           = 42

