"""
SFT (Supervised Fine-Tuning) com LoRA no GAIA.

Carrega CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it, aplica LoRA e treina em traces jurídicos.

Configs:
- LoRA rank: 64, alpha: 32
- QLoRA 4-bit
- Learning rate: 4e-4, cosine scheduler
- Batch 4, grad accum 4, 2 epochs
- Salva modelo + GGUF export

Usage:
    python -m training.sft.train_sft --config training/sft/config_sft.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer


# ============================================
# DEFAULT CONFIG
# ============================================

DEFAULT_CONFIG = {
    "model_name": "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    "max_seq_length": 8192,
    "load_in_4bit": True,
    "lora_r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "dataset_path": "training/data/sft_dataset/train.jsonl",
    "output_dir": "training/sft/checkpoints",
    "save_dir": "training/sft/gaia-legal-sft",
    "gguf_dir": "training/sft/gaia-legal-sft-gguf",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 4e-4,
    "num_train_epochs": 2,
    "warmup_steps": 50,
    "logging_steps": 10,
    "save_steps": 200,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "seed": 42,
}


# ============================================
# MODEL & LoRA
# ============================================

def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    """Load GAIA with FastLanguageModel and apply LoRA."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


# ============================================
# DATASET
# ============================================

def load_training_dataset(config: dict[str, Any]) -> Any:
    """Load dataset from JSONL (text field already formatted)."""
    ds_path = config["dataset_path"]
    if not Path(ds_path).exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {ds_path}\n"
            "Execute: python -m training.data.prepare_dataset"
        )
    dataset = load_dataset("json", data_files=ds_path, split="train")
    return dataset


# ============================================
# TRAINING
# ============================================

def train(config: dict[str, Any]) -> None:
    """Run SFT training."""
    print("Carregando modelo e tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    print("Carregando dataset...")
    dataset = load_training_dataset(config)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        optim=config["optim"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        seed=config["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )

    print("Iniciando treinamento...")
    trainer.train()

    # Save
    save_dir = config["save_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Modelo salvo em {save_dir}")

    # GGUF export
    gguf_dir = config.get("gguf_dir")
    if gguf_dir:
        Path(gguf_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"GGUF export salvo em {gguf_dir}")


# ============================================
# MAIN
# ============================================

def load_config(path: str | None) -> dict[str, Any]:
    """Load config from YAML or use defaults."""
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return {**DEFAULT_CONFIG, **loaded}
    return DEFAULT_CONFIG.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for GAIA Legal")
    parser.add_argument("--config", "-c", default="training/sft/config_sft.yaml", help="Config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
