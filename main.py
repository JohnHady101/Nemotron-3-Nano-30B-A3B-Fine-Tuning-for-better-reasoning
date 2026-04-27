import site

cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
site.addsitedir(cutlass_pkg_path)

# import traceback
import os
import shutil
import stat

# Patch the source before anything else
src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
dst = "/tmp/ptxas-blackwell"
shutil.copy2(src, dst)
os.chmod(dst, 0o755)

# Monkey-patch subprocess/os.execv to intercept the bad call
import subprocess
_orig_run = subprocess.run

def _patched_run(args, **kwargs):
    if isinstance(args, list) and "ptxas-blackwell" in str(args):
        args = [dst if "ptxas-blackwell" in a else a for a in args]
    return _orig_run(args, **kwargs)

subprocess.run = _patched_run

# Also patch os.execv family
import builtins
_orig_open = builtins.open

# Now trigger your error and capture the full traceback
try:
    # <<< paste whatever code causes the error here >>>
    pass
except PermissionError as e:
    traceback.print_exc()

import cutlass
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)
from sklearn.model_selection import train_test_split
import logging
import kagglehub


"""
LoRA Fine-tuning Pipeline for HuggingFace Transformer Models
Dataset format: Parquet with columns ['problem', 'thinking', 'solution']
No bitsandbytes dependency — runs in fp32, fp16, or bf16.
"""



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



from config import Config

cfg = Config()





# PROMPT FORMATTER

def build_prompt(example: dict) -> str:
    """
    Combine problem + thinking + solution into a single training string.
    The model learns to produce <think>…</think> then the final answer.
    """
    return (
        f"<|user|>\n{example['problem'].strip()}\n<|end|>\n"
        f"<|assistant|>\n"
        f"<think>\n{example['thinking'].strip()}\n</think>\n\n"
        f"{example['solution'].strip()}\n<|end|>"
    )



# LOAD & PREPARE DATASET


def load_dataset(path: str, tokenizer, val_split: float = 0.1):
    logger.info(f"Loading dataset from {path}")
    df = pd.read_parquet(path)

    required_cols = {"problem", "thinking", "solution"}
    assert required_cols.issubset(df.columns), (
        f"Parquet is missing columns: {required_cols - set(df.columns)}"
    )

    df = df.dropna(subset=list(required_cols)).reset_index(drop=True)
    logger.info(f"Dataset size after cleaning: {len(df):,} rows")

    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=cfg.SEED, shuffle=True
    )

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df,   preserve_index=False)

    def tokenize(example):
        prompt = build_prompt(example)
        encoded = tokenizer(
            prompt,
            max_length=cfg.MAX_LENGTH,
            truncation=cfg.TRUNCATION,
            padding=False,
        )
        # For causal LM: labels == input_ids (shifted inside the model)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    logger.info("Tokenising train split …")
    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)

    logger.info("Tokenising validation split …")
    val_ds = val_ds.map(tokenize, remove_columns=val_ds.column_names)

    return train_ds, val_ds



# LOAD MODEL + TOKENIZER

from transformers import BitsAndBytesConfig

def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_enable_fp32_cpu_offload=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if not cfg.USE_FP8 else None

    # FP8 via bitsandbytes (requires bnb >= 0.42 + Hopper/Ada GPU)
    fp8_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_has_fp16_weight=False,
        bnb_8bit_compute_dtype=torch.float8_e4m3fn,   # E4M3 format
        bnb_8bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        quantization_config=fp8_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Must call this before attaching LoRA to a quantized model
    model = prepare_model_for_kbit_training(model)




def load_model_and_tokenizer():
    logger.info(f"Loading tokenizer: {cfg.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.BASE_MODEL,
        trust_remote_code=True,
    )

    # Many models lack a pad token — reuse EOS as a safe default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"   # required for causal training

    logger.info(f"Loading base model: {cfg.BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        torch_dtype=cfg.TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False      # incompatible with gradient checkpointing
    model.enable_input_require_grads()  # needed for gradient checkpointing + PEFT

    return model, tokenizer



# ATTACH LORA ADAPTER


def attach_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=cfg.TARGET_MODULES,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model



# TRAINING ARGUMENTS


def build_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        num_train_epochs=cfg.NUM_EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        warmup_ratio=cfg.WARMUP_RATIO,
        lr_scheduler_type=cfg.LR_SCHEDULER,
        fp16=cfg.FP16,
        bf16=cfg.BF16,
        logging_dir=os.path.join(cfg.OUTPUT_DIR, "logs"),
        logging_steps=cfg.LOGGING_STEPS,
        eval_strategy="steps",          # fixed: uncommented + renamed from evaluation_strategy
        eval_steps=cfg.EVAL_STEPS,
        save_strategy="steps",
        save_steps=cfg.EVAL_STEPS,      # fixed: must equal eval_steps when load_best_model_at_end=True
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        gradient_checkpointing=True,
        # group_by_length=True,
        report_to="none",
        seed=cfg.SEED,
    )


# TRAIN


def train():
    # Load model & tokenizer 
    model, tokenizer = load_model_and_tokenizer()

    # Attach LoRA
    model = attach_lora(model)

    # Load & tokenise dataset
    train_ds, val_ds = load_dataset(cfg.DATASET_PATH, tokenizer, cfg.VAL_SPLIT)

    # Data collator
    # Pads sequences in each batch dynamically
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,    # -100 tells the loss to ignore pad tokens
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=build_training_args(),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("▶ Starting training …")
    trainer.train()

    # Save LoRA adapter only
    logger.info(f"Saving LoRA adapter to {cfg.OUTPUT_DIR}")
    trainer.model.save_pretrained(cfg.OUTPUT_DIR)
    tokenizer.save_pretrained(cfg.OUTPUT_DIR)

    logger.info("✅ Training complete.")
    return trainer



# MERGE ADAPTER INTO BASE MODEL (optional)

def merge_and_save():
    """
    Merge the LoRA weights back into the base model for efficient inference
    (no PEFT dependency needed at inference time).
    """
    logger.info("Loading base model for merge …")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        torch_dtype=cfg.TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapter from {cfg.OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, cfg.OUTPUT_DIR)

    logger.info("Merging adapter weights into base model …")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {cfg.MERGED_DIR}")
    model.save_pretrained(cfg.MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.OUTPUT_DIR)
    tokenizer.save_pretrained(cfg.MERGED_DIR)

    logger.info("✅ Merged model saved.")


# INFERENCE HELPER

def run_inference(problem: str, use_merged: bool = False, max_new_tokens: int = 512):
    """
    Quick inference check after training.
    Set use_merged=True to load from the merged model directory.
    """
    model_path = cfg.MERGED_DIR if use_merged else cfg.OUTPUT_DIR

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=cfg.TORCH_DTYPE,
        device_map="auto",
    )

    if not use_merged:
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    model.eval()

    prompt = f"<|user|>\n{problem.strip()}\n<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],   # strip the prompt tokens
        skip_special_tokens=True,
    )
    return response


if __name__ == "__main__":
    # Step 1 – Train
    trainer = train()

    # Step 2 – Merge adapter into base (optional, for deployment)
    merge_and_save()

    # Step 3 – Quick sanity check
    test_problem = "What is the derivative of x² + 3x + 5?"
    print("\n── Inference test ──────────────────────")
    print(f"Problem : {test_problem}")
    print(f"Response: {run_inference(test_problem, use_merged=True)}")

