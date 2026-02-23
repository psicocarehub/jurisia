"""
Quality filter for generated CoT traces.

Validates:
- Structure: proper <think> tags and sections
- Citations: article ranges via is_valid_article
- Language: consistent Portuguese (no English mixing)
- Length: minimum thinking/answer lengths
- Generates a quality report with metrics

Usage:
    python -m training.data.filter_quality --input raw_traces.jsonl --output filtered_traces.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


KNOWN_ARTICLE_RANGES: dict[str, int] = {
    "CF": 250, "CC": 2046, "CPC": 1072, "CLT": 922,
    "CDC": 119, "CTN": 185, "CP": 361, "CPP": 643, "ECA": 267,
}

ENGLISH_MARKERS = re.compile(
    r"\b(?:the|is|are|was|were|have|has|been|will|would|could|should|however|therefore|although)\b",
    re.IGNORECASE,
)

STRUCTURE_SECTIONS = re.compile(
    r"(?:IDENTIFICAÇÃO|LEGISLAÇÃO|JURISPRUDÊNCIA|ANÁLISE|CONCLUSÃO|ELIMINAÇÃO|FUNDAMENTAÇÃO)",
    re.IGNORECASE,
)

MIN_THINKING_CHARS = 100
MIN_ANSWER_CHARS = 20
MAX_ENGLISH_WORDS = 5


def validate_trace(trace: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a single trace. Returns (is_valid, list_of_issues).
    """
    issues: list[str] = []
    thinking = trace.get("thinking", "")
    answer = trace.get("answer", "")
    full = trace.get("full_response", "")

    if not thinking or len(thinking) < MIN_THINKING_CHARS:
        issues.append(f"thinking muito curto ({len(thinking)} chars, min {MIN_THINKING_CHARS})")

    if not answer or len(answer) < MIN_ANSWER_CHARS:
        issues.append(f"answer muito curta ({len(answer)} chars, min {MIN_ANSWER_CHARS})")

    if full and "<think>" not in full:
        issues.append("sem tag <think>")

    if thinking and not STRUCTURE_SECTIONS.search(thinking):
        issues.append("thinking sem seções estruturadas")

    text = thinking + " " + answer
    articles = re.findall(
        r"Art\.\s*(\d+)(?:[°ºª])?\s*(?:,?\s*d[aoe]\s+)?(CF|CC|CPC|CPP|CLT|CDC|CTN|CP|ECA)",
        text, re.IGNORECASE,
    )
    invalid_articles: list[str] = []
    for art_num, law in articles:
        law_upper = law.upper()
        if law_upper in KNOWN_ARTICLE_RANGES:
            try:
                if not (1 <= int(art_num) <= KNOWN_ARTICLE_RANGES[law_upper]):
                    invalid_articles.append(f"Art. {art_num} {law_upper}")
            except ValueError:
                invalid_articles.append(f"Art. {art_num} {law_upper}")

    if invalid_articles:
        issues.append(f"artigos inválidos: {', '.join(invalid_articles)}")

    english_count = len(ENGLISH_MARKERS.findall(text))
    if english_count > MAX_ENGLISH_WORDS:
        issues.append(f"mixing de idiomas ({english_count} palavras em inglês)")

    is_valid = len(issues) == 0

    if trace.get("pipeline") == "debate":
        citation_score = trace.get("citation_score", 1.0)
        if citation_score < 0.5:
            issues.append(f"citation_score baixo ({citation_score:.0%})")
            is_valid = False

    return is_valid, issues


def filter_traces(
    input_file: str,
    output_file: str,
    report_file: str | None = None,
    strict: bool = False,
) -> tuple[int, int, dict[str, Any]]:
    """
    Filter traces and generate quality report.

    Args:
        input_file: Input JSONL with raw traces
        output_file: Output JSONL with filtered traces
        report_file: Optional path for quality report JSON
        strict: If True, reject traces with any issue

    Returns:
        Tuple of (total, accepted, report_dict)
    """
    traces: list[dict[str, Any]] = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    issue_counts: dict[str, int] = defaultdict(int)
    area_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "accepted": 0})

    for trace in traces:
        area = trace.get("area", "geral")
        area_stats[area]["total"] += 1

        is_valid, issues = validate_trace(trace)

        if strict and issues:
            is_valid = False
        elif not strict and not is_valid:
            critical_issues = [
                i for i in issues
                if "muito curto" in i or "sem tag" in i or "mixing" in i
            ]
            is_valid = len(critical_issues) == 0

        for issue in issues:
            issue_type = issue.split("(")[0].strip().split(":")[0].strip()
            issue_counts[issue_type] += 1

        if is_valid:
            accepted.append(trace)
            area_stats[area]["accepted"] += 1
        else:
            rejected.append({**trace, "_rejection_reasons": issues})

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for t in accepted:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    accept_rate = len(accepted) / len(traces) * 100 if traces else 0
    report = {
        "total_traces": len(traces),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": round(accept_rate, 1),
        "issue_breakdown": dict(issue_counts),
        "area_stats": {
            area: {
                "total": stats["total"],
                "accepted": stats["accepted"],
                "rate": round(stats["accepted"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0,
            }
            for area, stats in sorted(area_stats.items())
        },
        "pipeline_stats": {
            "debate": sum(1 for t in accepted if t.get("pipeline") == "debate"),
            "simple": sum(1 for t in accepted if t.get("pipeline") != "debate"),
        },
    }

    print(f"\n=== Quality Report ===")
    print(f"Total: {report['total_traces']}")
    print(f"Aceitos: {report['accepted']} ({report['acceptance_rate']}%)")
    print(f"Rejeitados: {report['rejected']}")
    print(f"\nProblemas:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count}")
    print(f"\nPor área:")
    for area, stats in sorted(report["area_stats"].items()):
        print(f"  {area}: {stats['accepted']}/{stats['total']} ({stats['rate']}%)")

    if report_file:
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nRelatório salvo: {report_file}")

    return len(traces), len(accepted), report


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter CoT traces by quality")
    parser.add_argument("--input", "-i", default="training/data/raw_traces.jsonl")
    parser.add_argument("--output", "-o", default="training/data/filtered_traces.jsonl")
    parser.add_argument("--report", "-r", default="training/data/quality_report.json")
    parser.add_argument("--strict", action="store_true", help="Strict mode: reject any trace with issues")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Arquivo não encontrado: {args.input}")
        return

    filter_traces(args.input, args.output, args.report, args.strict)


if __name__ == "__main__":
    main()
