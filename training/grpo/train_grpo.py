"""
GRPO (Group Relative Policy Optimization) para reasoning jurídico.

Carrega modelo SFT, aplica GRPO com 4 reward functions:
- format_reward: <think>, estrutura argumentativa
- citation_reward: Art./Lei refs válidas
- correctness_reward: resposta correta
- language_reward: português consistente

Usage:
    python -m training.grpo.train_grpo --config training/grpo/config_grpo.yaml
"""

import argparse
from pathlib import Path
from typing import Any, Callable

import yaml
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from .rewards import (
    citation_reward,
    correctness_reward,
    format_reward,
    language_reward,
)


# ============================================
# DEFAULT CONFIG
# ============================================

DEFAULT_CONFIG = {
    "sft_model_path": "training/sft/gaia-legal-sft",
    "max_seq_length": 8192,
    "load_in_4bit": True,
    "dataset_path": "training/data/grpo_problems.jsonl",
    "output_dir": "training/grpo/checkpoints",
    "save_dir": "training/grpo/gaia-legal-reasoning",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-6,
    "num_generations": 8,
    "max_completion_length": 8192,
    "max_prompt_length": 2048,
    "temperature": 1.0,
    "beta": 0.001,
    "logging_steps": 5,
    "save_steps": 100,
    "bf16": True,
    "seed": 42,
}


# ============================================
# REWARD WRAPPERS (GRPO expects batch context)
# ============================================

def make_reward_func(
    base_func: Callable[..., list[float]],
    batch_key: str | None = None,
) -> Callable[..., list[float]]:
    """
    Wrap reward function to receive correct_answer from batch when needed.
    """

    def wrapped(completions: list[str], batch: dict[str, Any] | None = None, **kwargs: Any) -> list[float]:
        if batch and batch_key and batch_key in batch:
            kwargs["correct_answer"] = batch[batch_key]
        return base_func(completions, **kwargs)

    return wrapped


# ============================================
# MODEL & DATASET
# ============================================

def load_sft_model(config: dict[str, Any]) -> tuple[Any, Any]:
    """Load SFT model with Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["sft_model_path"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
    """Format example for GRPO (prompt + correct_answer)."""
    return {
        "prompt": [{"role": "user", "content": example["question"]}],
        "correct_answer": example.get("correct_answer", ""),
    }


# ============================================
# TRAINING
# ============================================

def train(config: dict[str, Any]) -> None:
    """Run GRPO training."""
    print("Carregando modelo SFT...")
    model, tokenizer = load_sft_model(config)

    ds_path = config["dataset_path"]
    if not Path(ds_path).exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {ds_path}\n"
            "Crie um JSONL com: {\"question\": \"...\", \"correct_answer\": \"A\" (opcional)}"
        )

    dataset = load_dataset("json", data_files=ds_path, split="train")
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    grpo_config = GRPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        max_prompt_length=config["max_prompt_length"],
        temperature=config["temperature"],
        beta=config["beta"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        bf16=config["bf16"],
        seed=config["seed"],
    )

    correctness_wrapper = make_reward_func(correctness_reward, "correct_answer")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,
            citation_reward,
            correctness_wrapper,
            language_reward,
        ],
        config=grpo_config,
        train_dataset=dataset,
    )

    print("Iniciando treinamento GRPO...")
    trainer.train()

    save_dir = config["save_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Modelo salvo em {save_dir}")


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
    parser = argparse.ArgumentParser(description="GRPO training for GAIA Legal")
    parser.add_argument("--config", "-c", default="training/grpo/config_grpo.yaml", help="Config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
