"""
Data Accumulator for continuous learning.

Collects new training data from multiple sources:
1. User feedback (corrections, positive examples)
2. New court decisions (auto-generated questions)
3. New legislation (generates Q&A pairs)
4. New sumulas and binding theses

When accumulated data reaches a configurable threshold,
signals readiness for a re-training cycle.

Usage:
    python -m training.continuous.data_accumulator --check
    python -m training.continuous.data_accumulator --collect
    python -m training.continuous.data_accumulator --export
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger("jurisai.accumulator")

DEFAULT_OUTPUT = "training/data/accumulated"
THRESHOLD_QUESTIONS = 5000
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


class DataAccumulator:
    """Accumulates training data from multiple sources."""

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT,
        threshold: int = THRESHOLD_QUESTIONS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self._state_file = self.output_dir / "accumulator_state.json"
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self._state_file.exists():
            return json.loads(self._state_file.read_text())
        return {
            "total_accumulated": 0,
            "sources": {},
            "last_collection": None,
            "last_export": None,
            "ready_for_training": False,
        }

    def _save_state(self) -> None:
        self._state_file.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2)
        )

    @property
    def is_ready(self) -> bool:
        return self._state["total_accumulated"] >= self.threshold

    @property
    def total_accumulated(self) -> int:
        return self._state["total_accumulated"]

    async def collect_from_feedback(self) -> int:
        """
        Collect training data from user feedback.

        - Edits become (query, corrected_response) pairs
        - Thumbs up + citation_used become positive examples
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning("Supabase not configured, skipping feedback collection")
            return 0

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/feedback_log",
                    headers=headers,
                    params={
                        "select": "*",
                        "processed": "eq.false",
                        "order": "created_at.asc",
                        "limit": "1000",
                    },
                )
                if resp.status_code != 200:
                    logger.warning("Failed to fetch feedback: %d", resp.status_code)
                    return 0
                feedback_rows = resp.json()
        except Exception as e:
            logger.error("Error fetching feedback: %s", e)
            return 0

        questions: list[dict[str, Any]] = []
        processed_ids: list[str] = []

        for row in feedback_rows:
            ft = row.get("feedback_type", "")
            query = row.get("query", "")
            fid = row.get("id", "")

            if ft == "edit" and row.get("edited_response") and query:
                questions.append({
                    "question": query,
                    "expected_answer": row["edited_response"],
                    "area": row.get("area", "geral"),
                    "source": "user_correction",
                    "difficulty": "medium",
                    "metadata": {"feedback_id": fid},
                })
                processed_ids.append(fid)

            elif ft in ("thumbs_up", "citation_used") and row.get("original_response") and query:
                questions.append({
                    "question": query,
                    "expected_answer": row["original_response"],
                    "area": row.get("area", "geral"),
                    "source": "user_approved",
                    "difficulty": "medium",
                    "metadata": {"feedback_id": fid},
                })
                processed_ids.append(fid)

        if questions:
            self._append_questions(questions, "feedback")

        if processed_ids:
            await self._mark_feedback_processed(processed_ids, headers)

        return len(questions)

    async def collect_from_decisions(self, days: int = 7) -> int:
        """
        Generate questions from recently ingested court decisions.
        Uses decision ementas to create open-ended legal questions.
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            return 0

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/ingestion_log",
                    headers=headers,
                    params={
                        "select": "source,records_count,created_at",
                        "status": "eq.completed",
                        "order": "created_at.desc",
                        "limit": "100",
                    },
                )
                if resp.status_code != 200:
                    return 0
                logs = resp.json()
        except Exception:
            return 0

        questions: list[dict[str, Any]] = []

        for log in logs:
            source = log.get("source", "")
            if not source.startswith("datajud_") and not source.startswith("stj_"):
                continue

            tribunal = source.replace("datajud_", "").replace("stj_", "").upper()

            questions.append({
                "question": f"Quais são as tendências recentes de decisão no {tribunal}?",
                "area": "processual",
                "source": "auto_decision",
                "difficulty": "hard",
                "metadata": {"tribunal": tribunal, "records": log.get("records_count", 0)},
            })

        if questions:
            self._append_questions(questions, "decisions")

        return len(questions)

    async def collect_from_legislation(self) -> int:
        """
        Generate Q&A pairs from newly ingested legislation (DOU docs).
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            return 0

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/dou_documents",
                    headers=headers,
                    params={
                        "select": "id,doc_type,title,ementa,publication_date",
                        "order": "ingested_at.desc",
                        "limit": "200",
                    },
                )
                if resp.status_code != 200:
                    return 0
                docs = resp.json()
        except Exception:
            return 0

        questions: list[dict[str, Any]] = []

        for doc in docs:
            title = doc.get("title", "")
            ementa = doc.get("ementa", "")
            doc_type = doc.get("doc_type", "")

            if not ementa or len(ementa) < 50:
                continue

            questions.extend([
                {
                    "question": f"O que dispõe a {title}?",
                    "expected_answer": ementa,
                    "area": "legislacao",
                    "source": "auto_legislation",
                    "difficulty": "medium",
                    "metadata": {"dou_id": doc.get("id"), "doc_type": doc_type},
                },
                {
                    "question": f"Quais são os principais pontos da {title} e como ela impacta a prática jurídica?",
                    "area": "legislacao",
                    "source": "auto_legislation",
                    "difficulty": "hard",
                    "metadata": {"dou_id": doc.get("id"), "doc_type": doc_type},
                },
            ])

        if questions:
            self._append_questions(questions, "legislation")

        return len(questions)

    async def collect_from_changes(self) -> int:
        """
        Generate training questions from detected law changes
        (revocations, amendments, new sumulas).
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            return 0

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/ingestion_log",
                    headers=headers,
                    params={
                        "select": "error_message,created_at",
                        "source": "eq.law_change",
                        "status": "eq.change_detected",
                        "order": "created_at.desc",
                        "limit": "200",
                    },
                )
                if resp.status_code != 200:
                    return 0
                changes = resp.json()
        except Exception:
            return 0

        questions: list[dict[str, Any]] = []

        for change in changes:
            msg = change.get("error_message", "")
            if not msg:
                continue

            if "REVOGACAO" in msg:
                questions.append({
                    "question": f"Houve alguma revogação recente relacionada a: {msg[11:200]}. Qual o impacto?",
                    "area": "legislacao",
                    "source": "auto_change",
                    "difficulty": "hard",
                    "metadata": {"change_type": "revocation"},
                })
            elif "ALTERACAO" in msg:
                questions.append({
                    "question": f"Uma alteração legislativa foi detectada: {msg[11:200]}. Quais são as implicações práticas?",
                    "area": "legislacao",
                    "source": "auto_change",
                    "difficulty": "hard",
                    "metadata": {"change_type": "amendment"},
                })

        if questions:
            self._append_questions(questions, "changes")

        return len(questions)

    async def collect_all(self) -> dict[str, int]:
        """Run all collection sources and return counts."""
        results = {
            "feedback": await self.collect_from_feedback(),
            "decisions": await self.collect_from_decisions(),
            "legislation": await self.collect_from_legislation(),
            "changes": await self.collect_from_changes(),
        }

        self._state["last_collection"] = datetime.now(timezone.utc).isoformat()
        self._state["ready_for_training"] = self.is_ready
        self._save_state()

        total = sum(results.values())
        logger.info(
            "Collected %d new items (total: %d / threshold: %d)",
            total, self.total_accumulated, self.threshold,
        )
        return results

    def export_for_training(self, output_path: Optional[str] = None) -> str:
        """
        Export all accumulated questions to a single JSONL file
        for the debate pipeline / CoT generation.
        """
        out = Path(output_path) if output_path else self.output_dir / "accumulated_questions.jsonl"

        all_questions: list[dict[str, Any]] = []
        for source_file in sorted(self.output_dir.glob("*.jsonl")):
            if source_file.name == "accumulated_questions.jsonl":
                continue
            with open(source_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_questions.append(json.loads(line))

        with open(out, "w", encoding="utf-8") as f:
            for q in all_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        self._state["last_export"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

        logger.info("Exported %d questions to %s", len(all_questions), out)
        return str(out)

    def _append_questions(
        self, questions: list[dict[str, Any]], source: str,
    ) -> None:
        """Append questions to the source-specific JSONL file."""
        out_file = self.output_dir / f"{source}.jsonl"
        with open(out_file, "a", encoding="utf-8") as f:
            for q in questions:
                q["accumulated_at"] = datetime.now(timezone.utc).isoformat()
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        prev = self._state["sources"].get(source, 0)
        self._state["sources"][source] = prev + len(questions)
        self._state["total_accumulated"] += len(questions)
        self._save_state()

    async def _mark_feedback_processed(
        self, feedback_ids: list[str], headers: dict[str, str],
    ) -> None:
        """Mark feedback entries as processed in Supabase."""
        for fid in feedback_ids:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.patch(
                        f"{SUPABASE_URL}/rest/v1/feedback_log",
                        headers={**headers, "Content-Type": "application/json", "Prefer": "return=minimal"},
                        params={"id": f"eq.{fid}"},
                        json={"processed": True},
                    )
            except Exception:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Accumulator for continuous learning")
    parser.add_argument("--check", action="store_true", help="Check accumulation status")
    parser.add_argument("--collect", action="store_true", help="Run collection from all sources")
    parser.add_argument("--export", action="store_true", help="Export accumulated data for training")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--threshold", type=int, default=THRESHOLD_QUESTIONS, help="Training threshold")
    args = parser.parse_args()

    acc = DataAccumulator(output_dir=args.output_dir, threshold=args.threshold)

    if args.check:
        print(f"Total accumulated: {acc.total_accumulated}")
        print(f"Threshold: {acc.threshold}")
        print(f"Ready for training: {acc.is_ready}")
        print(f"Sources: {json.dumps(acc._state.get('sources', {}), indent=2)}")
        print(f"Last collection: {acc._state.get('last_collection', 'never')}")
        return

    if args.collect:
        results = asyncio.run(acc.collect_all())
        print(f"Collection results: {json.dumps(results, indent=2)}")
        print(f"Total accumulated: {acc.total_accumulated} / {acc.threshold}")
        if acc.is_ready:
            print("READY FOR TRAINING!")
        return

    if args.export:
        path = acc.export_for_training()
        print(f"Exported to: {path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
