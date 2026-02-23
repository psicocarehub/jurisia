"""
Coleta de corpus para continued pre-training do GAIA.

Baixa e filtra:
- Multi_Legal_Pile (joelniklaus/Multi_Legal_Pile) — subsets PT: caselaw, legislation, contracts
- GigaVerbo (TucanoBR/GigaVerbo) — textos PT filtrados por keywords juridicos

Deduplicacao via MinHash-LSH. Salva em formato tokenizado para pre-training.

Usage:
    python -m training.data.collect_pretraining -o training/data/pretraining_corpus
    python -m training.data.collect_pretraining --skip-gigaverbo -o training/data/pretraining_corpus
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("jurisai.pretraining")

MULTI_LEGAL_CONFIGS = ["pt_caselaw", "pt_legislation", "pt_contracts", "pt_other"]

LEGAL_KEYWORDS = [
    "código", "lei", "decreto", "artigo", "contrato", "cláusula", "tribunal",
    "jurisprudência", "sentença", "acórdão", "recurso", "advogado", "juiz",
    "ministério público", "réu", "autor", "processo", "ação", "petição",
    "mandado", "habeas corpus", "liminar", "tutela", "execução", "pena",
    "crime", "delito", "infração", "constituição", "emenda", "súmula",
    "voto", "relator", "desembargador", "ministro", "procurador",
    "direito", "obrigação", "responsabilidade", "indenização", "dano",
    "trabalhista", "previdenciário", "tributário", "administrativo",
    "licitação", "imposto", "contribuição", "ICMS", "ISS", "IRPF",
]


def _text_is_legal(text: str, min_keywords: int = 3) -> bool:
    """Check if text contains enough legal keywords."""
    text_lower = text.lower()
    count = sum(1 for kw in LEGAL_KEYWORDS if kw.lower() in text_lower)
    return count >= min_keywords


def collect_multi_legal_pile(output_dir: Path, max_per_config: int = 200_000) -> int:
    """Download and filter Multi_Legal_Pile PT subsets."""
    from datasets import load_dataset

    total = 0
    for config in MULTI_LEGAL_CONFIGS:
        out_file = output_dir / f"multi_legal_{config}.jsonl"
        if out_file.exists():
            logger.info("Pulando %s (ja existe)", config)
            existing = sum(1 for _ in open(out_file))
            total += existing
            continue

        logger.info("Baixando Multi_Legal_Pile config=%s ...", config)
        try:
            ds = load_dataset(
                "joelniklaus/Multi_Legal_Pile",
                config,
                split="train",
                streaming=True,
            )
        except Exception as e:
            logger.warning("Erro carregando %s: %s", config, e)
            continue

        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for example in ds:
                text = example.get("text", "")
                if len(text) < 200:
                    continue
                f.write(json.dumps({"text": text, "source": f"multi_legal_{config}"}, ensure_ascii=False) + "\n")
                count += 1
                if count >= max_per_config:
                    break
                if count % 10_000 == 0:
                    logger.info("  %s: %d textos", config, count)

        total += count
        logger.info("  %s: %d textos salvos", config, count)

    return total


def collect_gigaverbo(output_dir: Path, max_texts: int = 200_000) -> int:
    """Download GigaVerbo and filter for legal texts."""
    from datasets import load_dataset

    out_file = output_dir / "gigaverbo_legal.jsonl"
    if out_file.exists():
        logger.info("Pulando GigaVerbo (ja existe)")
        return sum(1 for _ in open(out_file))

    logger.info("Baixando GigaVerbo (filtrado por keywords juridicos)...")
    try:
        ds = load_dataset("TucanoBR/GigaVerbo", split="train", streaming=True)
    except Exception as e:
        logger.warning("Erro carregando GigaVerbo: %s", e)
        return 0

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", example.get("content", ""))
            if len(text) < 300:
                continue
            if not _text_is_legal(text):
                continue
            f.write(json.dumps({"text": text, "source": "gigaverbo_legal"}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_texts:
                break
            if count % 5_000 == 0:
                logger.info("  GigaVerbo legal: %d textos", count)

    logger.info("GigaVerbo legal: %d textos salvos", count)
    return count


def deduplicate_corpus(output_dir: Path, threshold: float = 0.85) -> int:
    """Deduplicate all corpus files using MinHash-LSH."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "api"))

    from app.services.ingestion.deduplicator import LSHIndex, MinHash

    logger.info("Deduplicando corpus com MinHash-LSH (threshold=%.2f)...", threshold)
    lsh = LSHIndex(threshold=threshold)
    seen_hashes: dict[str, int] = {}
    doc_id = 0

    all_files = sorted(output_dir.glob("*.jsonl"))
    dedup_file = output_dir / "corpus_dedup.jsonl"

    kept = 0
    total = 0

    with open(dedup_file, "w", encoding="utf-8") as out:
        for fpath in all_files:
            if fpath.name == "corpus_dedup.jsonl":
                continue
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    total += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text", "")
                    mh = MinHash(text)
                    if lsh.is_duplicate(mh):
                        continue
                    lsh.insert(doc_id, mh)
                    doc_id += 1
                    out.write(line)
                    kept += 1

                    if total % 50_000 == 0:
                        logger.info("  Dedup: %d/%d mantidos", kept, total)

    logger.info("Deduplicacao: %d -> %d textos (%.1f%% removidos)", total, kept, (1 - kept / max(total, 1)) * 100)
    return kept


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Collect pre-training corpus")
    parser.add_argument("--output", "-o", default="training/data/pretraining_corpus")
    parser.add_argument("--max-per-source", type=int, default=200_000)
    parser.add_argument("--skip-gigaverbo", action="store_true")
    parser.add_argument("--skip-dedup", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    total += collect_multi_legal_pile(output_dir, max_per_config=args.max_per_source)

    if not args.skip_gigaverbo:
        total += collect_gigaverbo(output_dir, max_texts=args.max_per_source)

    if not args.skip_dedup and total > 0:
        deduplicate_corpus(output_dir)

    logger.info("Corpus total: %d textos em %s", total, output_dir)


if __name__ == "__main__":
    main()
