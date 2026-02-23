"""
Incremental fine-tuning pipeline for GAIA.

Instead of training from scratch each cycle, this script:
1. Loads the current best model (with merged LoRA weights)
2. Applies a new LoRA adapter on top
3. Trains on accumulated new data (smaller dataset, fewer epochs)
4. Evaluates on OAB benchmark
5. Compares against the current production model
6. If improved: merges LoRA, saves as new version, deploys with A/B flag
7. If degraded: discards and logs for investigation

Usage:
    python -m training.continuous.incremental_finetune --check
    python -m training.continuous.incremental_finetune --train
    python -m training.continuous.incremental_finetune --compare
"""

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("jurisai.incremental_ft")

VERSIONS_DIR = "training/continuous/versions"
EVAL_RESULTS_DIR = "training/continuous/eval_results"
ACCUMULATED_DATA = "training/data/accumulated/accumulated_questions.jsonl"

DEFAULT_CONFIG = {
    "base_model": "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    "current_adapter": "training/sft/gaia-legal-sft",
    "max_seq_length": 8192,
    "load_in_4bit": True,
    "lora_r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "warmup_steps": 20,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "seed": 42,
    "min_improvement": 0.02,
    "ab_test_ratio": 0.1,
}


class VersionManager:
    """Manages model versions for A/B testing and rollback."""

    def __init__(self, versions_dir: str = VERSIONS_DIR) -> None:
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.versions_dir / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {
            "current_version": None,
            "candidate_version": None,
            "versions": [],
            "ab_test_active": False,
            "ab_test_ratio": 0.1,
        }

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=2)
        )

    @property
    def current_version(self) -> Optional[str]:
        return self._manifest.get("current_version")

    @property
    def candidate_version(self) -> Optional[str]:
        return self._manifest.get("candidate_version")

    @property
    def ab_test_active(self) -> bool:
        return self._manifest.get("ab_test_active", False)

    def register_version(
        self,
        version_id: str,
        model_path: str,
        metrics: dict[str, float],
        training_data_count: int,
    ) -> None:
        """Register a new model version."""
        entry = {
            "version_id": version_id,
            "model_path": model_path,
            "metrics": metrics,
            "training_data_count": training_data_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "candidate",
        }
        self._manifest["versions"].append(entry)
        self._manifest["candidate_version"] = version_id
        self._save_manifest()

    def promote_candidate(self) -> None:
        """Promote current candidate to production."""
        candidate = self._manifest.get("candidate_version")
        if not candidate:
            return

        for v in self._manifest["versions"]:
            if v["version_id"] == self._manifest.get("current_version"):
                v["status"] = "retired"
            if v["version_id"] == candidate:
                v["status"] = "production"

        self._manifest["current_version"] = candidate
        self._manifest["candidate_version"] = None
        self._manifest["ab_test_active"] = False
        self._save_manifest()
        logger.info("Promoted %s to production", candidate)

    def reject_candidate(self, reason: str = "") -> None:
        """Reject candidate and discard."""
        candidate = self._manifest.get("candidate_version")
        if not candidate:
            return

        for v in self._manifest["versions"]:
            if v["version_id"] == candidate:
                v["status"] = "rejected"
                v["rejection_reason"] = reason

        self._manifest["candidate_version"] = None
        self._manifest["ab_test_active"] = False
        self._save_manifest()
        logger.info("Rejected candidate %s: %s", candidate, reason)

    def start_ab_test(self, ratio: float = 0.1) -> None:
        """Start A/B test between current and candidate."""
        if not self._manifest.get("candidate_version"):
            logger.warning("No candidate to A/B test")
            return
        self._manifest["ab_test_active"] = True
        self._manifest["ab_test_ratio"] = ratio
        self._save_manifest()
        logger.info("A/B test started: %.0f%% traffic to candidate", ratio * 100)

    def get_model_for_request(self, random_val: float) -> str:
        """
        Returns model path based on A/B test ratio.
        random_val should be uniform [0, 1).
        """
        if self.ab_test_active and self.candidate_version:
            ratio = self._manifest.get("ab_test_ratio", 0.1)
            if random_val < ratio:
                for v in self._manifest["versions"]:
                    if v["version_id"] == self.candidate_version:
                        return v["model_path"]

        if self.current_version:
            for v in self._manifest["versions"]:
                if v["version_id"] == self.current_version:
                    return v["model_path"]

        return DEFAULT_CONFIG["current_adapter"]

    def get_status(self) -> dict[str, Any]:
        return {
            "current_version": self.current_version,
            "candidate_version": self.candidate_version,
            "ab_test_active": self.ab_test_active,
            "total_versions": len(self._manifest["versions"]),
            "versions": self._manifest["versions"][-5:],
        }


def incremental_train(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Run incremental fine-tuning on accumulated data.

    Returns metrics dict with training results.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    data_path = Path(ACCUMULATED_DATA)
    if not data_path.exists():
        logger.error("No accumulated data found at %s", data_path)
        return {"error": "no_data"}

    with open(data_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 100:
        logger.warning("Only %d samples — too few for incremental training", len(lines))
        return {"error": "insufficient_data", "count": len(lines)}

    version_id = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"{VERSIONS_DIR}/{version_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting incremental training: %s (%d samples)", version_id, len(lines))

    try:
        import torch
        from datasets import load_dataset
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel
        from trl import SFTTrainer

        base_model = cfg.get("current_adapter") or cfg["base_model"]

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=cfg["max_seq_length"],
            dtype=None,
            load_in_4bit=cfg["load_in_4bit"],
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            target_modules=cfg["target_modules"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        dataset = load_dataset("json", data_files=str(data_path), split="train")

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/checkpoints",
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=cfg["warmup_steps"],
            num_train_epochs=cfg["num_train_epochs"],
            learning_rate=cfg["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            save_strategy="epoch",
            optim=cfg["optim"],
            weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            seed=cfg["seed"],
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
            args=training_args,
        )

        trainer.train()

        save_path = f"{output_dir}/model"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        train_loss = trainer.state.log_history[-1].get("train_loss", 0.0)
        logger.info("Training complete: loss=%.4f, saved to %s", train_loss, save_path)

        return {
            "version_id": version_id,
            "model_path": save_path,
            "train_loss": train_loss,
            "data_count": len(lines),
            "epochs": cfg["num_train_epochs"],
        }

    except ImportError as e:
        logger.error("Training dependencies not available: %s", e)
        return {"error": "missing_deps", "detail": str(e)}
    except Exception as e:
        logger.error("Training failed: %s", e)
        return {"error": "training_failed", "detail": str(e)}


def compare_models(
    current_path: str,
    candidate_path: str,
    questions_path: str = "training/data/oab_questions.jsonl",
) -> dict[str, Any]:
    """
    Compare current and candidate models on OAB benchmark.

    Returns comparison dict with metrics and recommendation.
    """
    from training.eval.eval_oab import run_eval

    eval_dir = Path(EVAL_RESULTS_DIR)
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info("Evaluating current model: %s", current_path)
    current_metrics = run_eval(
        model_path=current_path,
        questions_path=questions_path,
        output_path=str(eval_dir / f"current_{timestamp}.json"),
    )

    logger.info("Evaluating candidate model: %s", candidate_path)
    candidate_metrics = run_eval(
        model_path=candidate_path,
        questions_path=questions_path,
        output_path=str(eval_dir / f"candidate_{timestamp}.json"),
    )

    current_acc = current_metrics.get("accuracy", 0.0)
    candidate_acc = candidate_metrics.get("accuracy", 0.0)
    improvement = candidate_acc - current_acc

    min_improvement = DEFAULT_CONFIG["min_improvement"]
    recommendation = "promote" if improvement >= min_improvement else "reject"

    area_comparison = {}
    for key in candidate_metrics:
        if key.startswith("accuracy_"):
            area = key.replace("accuracy_", "")
            curr = current_metrics.get(key, 0.0)
            cand = candidate_metrics.get(key, 0.0)
            area_comparison[area] = {
                "current": round(curr, 4),
                "candidate": round(cand, 4),
                "delta": round(cand - curr, 4),
            }

    degraded_areas = [
        area for area, data in area_comparison.items()
        if data["delta"] < -0.05
    ]
    if degraded_areas and recommendation == "promote":
        recommendation = "ab_test"

    result = {
        "current_accuracy": round(current_acc, 4),
        "candidate_accuracy": round(candidate_acc, 4),
        "improvement": round(improvement, 4),
        "recommendation": recommendation,
        "degraded_areas": degraded_areas,
        "area_comparison": area_comparison,
        "timestamp": timestamp,
    }

    result_path = eval_dir / f"comparison_{timestamp}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info(
        "Comparison: current=%.2f%%, candidate=%.2f%%, delta=%.2f%%, recommendation=%s",
        current_acc * 100, candidate_acc * 100, improvement * 100, recommendation,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental fine-tuning for GAIA")
    parser.add_argument("--check", action="store_true", help="Check version status")
    parser.add_argument("--train", action="store_true", help="Run incremental training")
    parser.add_argument("--compare", action="store_true", help="Compare current vs candidate")
    parser.add_argument("--promote", action="store_true", help="Promote candidate to production")
    parser.add_argument("--reject", action="store_true", help="Reject candidate")
    parser.add_argument("--ab-test", action="store_true", help="Start A/B test")
    parser.add_argument("--ab-ratio", type=float, default=0.1, help="A/B test traffic ratio")
    args = parser.parse_args()

    vm = VersionManager()

    if args.check:
        status = vm.get_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    if args.train:
        result = incremental_train()
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if "error" not in result:
            vm.register_version(
                version_id=result["version_id"],
                model_path=result["model_path"],
                metrics={"train_loss": result["train_loss"]},
                training_data_count=result["data_count"],
            )
            print(f"\nVersion {result['version_id']} registered as candidate")
            print("Run --compare to evaluate against production")
        return

    if args.compare:
        if not vm.current_version and not vm.candidate_version:
            print("No versions to compare")
            return

        current_path = DEFAULT_CONFIG["current_adapter"]
        candidate_path = None

        for v in vm._manifest["versions"]:
            if v["version_id"] == vm.current_version:
                current_path = v["model_path"]
            if v["version_id"] == vm.candidate_version:
                candidate_path = v["model_path"]

        if not candidate_path:
            print("No candidate version to compare")
            return

        result = compare_models(current_path, candidate_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if result["recommendation"] == "promote":
            print("\nRecommendation: PROMOTE candidate to production")
        elif result["recommendation"] == "ab_test":
            print("\nRecommendation: Run A/B test (some areas degraded)")
        else:
            print("\nRecommendation: REJECT candidate (insufficient improvement)")
        return

    if args.promote:
        vm.promote_candidate()
        print("Candidate promoted to production")
        return

    if args.reject:
        vm.reject_candidate("Manual rejection")
        print("Candidate rejected")
        return

    if args.ab_test:
        vm.start_ab_test(ratio=args.ab_ratio)
        print(f"A/B test started with {args.ab_ratio:.0%} traffic to candidate")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
