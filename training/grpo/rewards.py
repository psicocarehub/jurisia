"""
Reward functions para GRPO (Group Relative Policy Optimization).

Funções rule-based para reforçar:
1. Format: tags <think>, estrutura argumentativa
2. Citation: artigos/leis válidos
3. Correctness: resposta correta quando verificável
4. Language: português consistente (sem mixing)
"""

import re
from typing import Any


# Known article ranges (simplified — expand with real legislation DB)
KNOWN_ARTICLE_RANGES: dict[str, int] = {
    "CF": 250,   # Constituição Federal
    "CC": 2046,  # Código Civil
    "CPC": 1072, # Código de Processo Civil
    "CLT": 922,  # Consolidação das Leis do Trabalho
    "CDC": 119,  # Código de Defesa do Consumidor
    "CTN": 185,  # Código Tributário Nacional
    "CP": 361,   # Código Penal
    "CPP": 643,  # Código de Processo Penal
    "ECA": 267,  # Estatuto da Criança e do Adolescente
}


def is_valid_article(art_num: str, law: str, law_num: str) -> bool:
    """
    Check if article exists in known legislation (simplified).

    Args:
        art_num: Article number
        law: Law identifier (CF, CC, CPC, etc.)
        law_num: Optional law number (e.g. "10.406/2002")

    Returns:
        True if article is in valid range or law is unknown (assume valid)
    """
    law_upper = law.upper()
    if law_upper in KNOWN_ARTICLE_RANGES:
        try:
            return 1 <= int(art_num) <= KNOWN_ARTICLE_RANGES[law_upper]
        except ValueError:
            return False
    return True  # Unknown law, assume valid


def format_reward(completions: list[str], **kwargs: Any) -> list[float]:
    """
    Reward for proper reasoning format (<think>, estrutura).

    Args:
        completions: List of model completion strings

    Returns:
        List of reward scores (0.0 to ~0.5)
    """
    rewards: list[float] = []
    for text in completions:
        score = 0.0

        if "<think>" in text and "</think>" in text:
            score += 0.2
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
            if think_match:
                content = think_match.group(1)
                if re.search(
                    r"(?:IDENTIFICAÇÃO|LEGISLAÇÃO|JURISPRUDÊNCIA|ANÁLISE|CONCLUSÃO|ELIMINAÇÃO)",
                    content,
                    re.IGNORECASE,
                ):
                    score += 0.2
                if re.search(r"Art\.\s*\d+", content):
                    score += 0.1

        rewards.append(score)
    return rewards


def citation_reward(completions: list[str], **kwargs: Any) -> list[float]:
    """
    Reward for correct legal citations (Art., Lei, Súmula).

    Args:
        completions: List of model completion strings

    Returns:
        List of reward scores (0.0 to 2.0)
    """
    rewards: list[float] = []
    for text in completions:
        score = 0.0

        articles = re.findall(
            r"Art\.\s*(\d+)(?:[°ºª])?\s*(?:,?\s*d[aoe]\s+)?(Lei|CF|CC|CPC|CPP|CLT|CDC|CTN|CP|ECA)\s*(?:n[ºo°]?\s*)?([\d./]*)",
            text,
            re.IGNORECASE,
        )

        if articles:
            score += min(len(articles) * 0.3, 1.5)
            for art_num, law, law_num in articles:
                if is_valid_article(art_num, law, law_num):
                    score += 0.1

        sumulas = re.findall(r"Súmula\s+(?:Vinculante\s+)?(\d+)", text)
        if sumulas:
            score += min(len(sumulas) * 0.2, 0.5)

        rewards.append(min(score, 2.0))
    return rewards


def correctness_reward(
    completions: list[str],
    correct_answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Reward for correct final answer (when verifiable).

    Args:
        completions: List of model completion strings
        correct_answer: List of correct answers (one per prompt)

    Returns:
        List of reward scores (0.0 or 2.0)
    """
    if correct_answer is None:
        return [0.0] * len(completions)

    rewards: list[float] = []
    for completion, answer in zip(completions, correct_answer):
        if not answer:
            rewards.append(0.0)
            continue

        if "</think>" in completion:
            response = completion.split("</think>", 1)[1].strip()
        else:
            response = completion.strip()

        pred = response[:100].strip().upper()
        gold = answer.strip().upper()

        if gold in pred or pred.startswith(gold):
            rewards.append(2.0)
        else:
            rewards.append(0.0)

    return rewards


def language_reward(completions: list[str], **kwargs: Any) -> list[float]:
    """
    Reward for consistent Portuguese (no English mixing).

    Args:
        completions: List of model completion strings

    Returns:
        List of reward scores (-0.5 to 0.3)
    """
    rewards: list[float] = []
    english_markers = re.compile(
        r"\b(?:the|is|are|was|were|have|has|been|will|would|could|should)\b",
        re.IGNORECASE,
    )
    for text in completions:
        count = len(english_markers.findall(text))
        if count > 5:
            rewards.append(-0.5)
        elif count > 0:
            rewards.append(0.0)
        else:
            rewards.append(0.3)
    return rewards
