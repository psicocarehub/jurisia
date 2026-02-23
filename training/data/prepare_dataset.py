"""
Preparar dataset para SFT a partir de traces JSONL.

Carrega traces gerados por generate_cot.py, formata no template Gemma chat,
faz split train/val e salva em formato compatível com SFTTrainer.

Usage:
    python -m training.data.prepare_dataset --input traces.jsonl --output training/data/sft_dataset
"""

import argparse
import json
from pathlib import Path
from typing import Any

# ============================================
# GEMMA CHAT TEMPLATE
# ============================================

CHAT_TEMPLATE = """<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
<think>
{thinking}
</think>

{answer}<end_of_turn>"""


def format_example(example: dict[str, Any]) -> dict[str, str]:
    """
    Format a trace into Gemma chat format.

    Args:
        example: Dict with question, thinking, answer

    Returns:
        Dict with 'text' and optionally 'messages'
    """
    return {
        "text": CHAT_TEMPLATE.format(
            question=example["question"],
            thinking=example["thinking"],
            answer=example["answer"],
        ),
    }


def load_traces(path: str) -> list[dict[str, Any]]:
    """Load traces from JSONL file."""
    traces: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def prepare_dataset(
    input_file: str,
    output_dir: str,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Load traces, format, split train/val, and save.

    Args:
        input_file: Input JSONL path
        output_dir: Output directory for train/val JSONL files
        val_ratio: Fraction for validation (0.05 = 5%)
        seed: Random seed for split

    Returns:
        Tuple of (num_train, num_val)
    """
    import random

    traces = load_traces(input_file)
    if not traces:
        raise ValueError(f"Nenhum trace encontrado em {input_file}")

    formatted = [format_example(t) for t in traces]
    random.seed(seed)
    random.shuffle(formatted)

    n_val = max(1, int(len(formatted) * val_ratio))
    n_train = len(formatted) - n_val
    train_data = formatted[:n_train]
    val_data = formatted[n_train:]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset preparado: {n_train} train, {n_val} val")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    return n_train, n_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT dataset from traces")
    parser.add_argument("--input", "-i", default="training/data/raw_traces.jsonl", help="Input traces JSONL")
    parser.add_argument("--output", "-o", default="training/data/sft_dataset", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Arquivo não encontrado: {args.input}")
        print("Execute primeiro: python -m training.data.generate_cot")
        return

    prepare_dataset(
        input_file=args.input,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
