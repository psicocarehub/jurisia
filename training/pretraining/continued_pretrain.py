"""
Continued pre-training do GAIA 4B no corpus juridico brasileiro.

Usa LoRA para eficiencia de memoria. O corpus vem de Multi_Legal_Pile + GigaVerbo
(filtrado e deduplicado por collect_pretraining.py).

Ordem do pipeline completo:
  1. Continued pre-training (este script) — adapta o modelo ao dominio juridico
  2. SFT com CoT traces (train_sft.py) — ensina raciocinio passo-a-passo
  3. GRPO (train_grpo.py) — alinhamento por recompensa

Usage:
    python -m training.pretraining.continued_pretrain \\
        --corpus-dir training/data/pretraining_corpus \\
        --output-dir training/checkpoints/pretrained \\
        --epochs 2 --lr 1e-5
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("jurisai.pretrain")


def load_corpus(corpus_dir: str, max_length: int = 2048) -> list[str]:
    """Load deduplicated corpus, preferring the dedup file."""
    corpus_path = Path(corpus_dir)
    dedup_file = corpus_path / "corpus_dedup.jsonl"
    source_files = sorted(corpus_path.glob("*.jsonl"))

    target = dedup_file if dedup_file.exists() else None
    if target is None:
        source_files = [f for f in source_files if f.name != "corpus_dedup.jsonl"]

    texts: list[str] = []
    files = [target] if target else source_files

    for fpath in files:
        if fpath is None:
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if len(text) >= 100:
                    texts.append(text[:max_length * 4])

    logger.info("Corpus carregado: %d textos", len(texts))
    return texts


def train(
    corpus_dir: str = "training/data/pretraining_corpus",
    output_dir: str = "training/checkpoints/pretrained",
    model_name: str = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    epochs: int = 2,
    lr: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation: int = 8,
    max_seq_len: int = 2048,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    use_unsloth: bool = True,
) -> None:
    """Run continued pre-training with LoRA on the legal corpus."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    texts = load_corpus(corpus_dir, max_length=max_seq_len)
    if not texts:
        logger.error("Corpus vazio em %s", corpus_dir)
        return

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            logger.warning("Unsloth nao instalado, usando HuggingFace padrao")
            use_unsloth = False

    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=lora_target_modules.split(","),
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules.split(","),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, TrainingArguments
    from trl import SFTTrainer

    dataset = Dataset.from_dict({"text": texts})

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        optim="adamw_8bit" if use_unsloth else "adamw_torch",
        report_to="none",
        dataloader_num_workers=4,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
    )

    logger.info("Iniciando continued pre-training: %d textos, %d epochs, lr=%.1e", len(texts), epochs, lr)
    trainer.train()

    final_dir = str(Path(output_dir) / "final")
    if use_unsloth:
        model.save_pretrained_merged(final_dir, tokenizer, save_method="merged_16bit")
    else:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

    logger.info("Modelo salvo em %s", final_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Continued pre-training of GAIA on legal corpus")
    parser.add_argument("--corpus-dir", default="training/data/pretraining_corpus")
    parser.add_argument("--output-dir", default="training/checkpoints/pretrained")
    parser.add_argument("--model", default="CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--no-unsloth", action="store_true")
    args = parser.parse_args()

    train(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_unsloth=not args.no_unsloth,
    )


if __name__ == "__main__":
    main()
