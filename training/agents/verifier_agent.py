"""
Verifier Agent: checks all legal citations in the jurist's response.

Uses rule-based extraction (regex) + is_valid_article from rewards.py
to verify article numbers, sumula references, and law citations.
"""

import re
from typing import Any

from training.agents.state import CitationCheck, DebateState

KNOWN_ARTICLE_RANGES: dict[str, int] = {
    "CF": 250,
    "CC": 2046,
    "CPC": 1072,
    "CLT": 922,
    "CDC": 119,
    "CTN": 185,
    "CP": 361,
    "CPP": 643,
    "ECA": 267,
    "LEA": 38,   # Lei de Execução Penal (LEP) — simplified
    "LINDB": 19,
}

KNOWN_SUMULA_RANGES: dict[str, int] = {
    "STF": 736,
    "STJ": 660,
    "TST": 463,
    "TSE": 73,
}

VINCULANTE_MAX = 58


def _is_valid_article(art_num: str, law: str) -> bool:
    law_upper = law.upper().strip()
    if law_upper in KNOWN_ARTICLE_RANGES:
        try:
            return 1 <= int(art_num) <= KNOWN_ARTICLE_RANGES[law_upper]
        except ValueError:
            return False
    return True


def _is_valid_sumula(num: str, tribunal: str, is_vinculante: bool) -> bool:
    try:
        n = int(num)
    except ValueError:
        return False

    if is_vinculante:
        return 1 <= n <= VINCULANTE_MAX

    tribunal_upper = tribunal.upper().strip()
    if tribunal_upper in KNOWN_SUMULA_RANGES:
        return 1 <= n <= KNOWN_SUMULA_RANGES[tribunal_upper]

    return True


async def verifier_node(state: DebateState) -> dict:
    """Verify all legal citations in the jurist's response."""
    text = state.get("jurist_response", "")
    if not text:
        return {
            "citation_checks": [],
            "citation_score": 0.0,
            "citation_feedback": "Sem resposta do jurista para verificar.",
        }

    checks: list[CitationCheck] = []

    articles = re.findall(
        r"Art\.\s*(\d+)(?:[°ºª])?\s*(?:,?\s*d[aoe]\s+)?(Lei|CF|CC|CPC|CPP|CLT|CDC|CTN|CP|ECA|LEA|LINDB)\s*(?:n[ºo°]?\s*)?([\d./]*)",
        text, re.IGNORECASE,
    )
    for art_num, law, law_num in articles:
        valid = _is_valid_article(art_num, law)
        issue = "" if valid else f"Art. {art_num} fora do range conhecido para {law.upper()} (max: {KNOWN_ARTICLE_RANGES.get(law.upper(), '?')})"
        checks.append(CitationCheck(
            text=f"Art. {art_num} {law} {law_num}".strip(),
            type="legislacao",
            valid=valid,
            issue=issue,
        ))

    sumulas = re.findall(
        r"Súmula\s+(Vinculante\s+)?(?:n[ºo°]?\s*)?(\d+)\s*(?:d[oe]\s+(STF|STJ|TST|TSE))?",
        text, re.IGNORECASE,
    )
    for vinc, num, tribunal in sumulas:
        is_vinculante = bool(vinc and vinc.strip())
        trib = tribunal.upper() if tribunal else ("STF" if is_vinculante else "")
        valid = _is_valid_sumula(num, trib, is_vinculante)
        label = f"Súmula {'Vinculante ' if is_vinculante else ''}{num} {trib}".strip()
        issue = "" if valid else f"{label} fora do range conhecido"
        checks.append(CitationCheck(
            text=label,
            type="sumula",
            valid=valid,
            issue=issue,
        ))

    cnj_refs = re.findall(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}", text)
    for ref in cnj_refs:
        checks.append(CitationCheck(
            text=ref,
            type="jurisprudencia",
            valid=True,
            issue="",
        ))

    total = len(checks)
    valid_count = sum(1 for c in checks if c["valid"])
    score = valid_count / total if total > 0 else 1.0

    issues = [c for c in checks if not c["valid"]]
    if issues:
        feedback_lines = ["CITAÇÕES COM PROBLEMAS:"]
        for c in issues:
            feedback_lines.append(f"  - {c['text']}: {c['issue']}")
        feedback = "\n".join(feedback_lines)
    elif total > 0:
        feedback = f"Todas as {total} citações verificadas com sucesso."
    else:
        feedback = "Nenhuma citação legal encontrada na resposta. Considere incluir fundamentação legal."

    return {
        "citation_checks": checks,
        "citation_score": score,
        "citation_feedback": feedback,
    }
