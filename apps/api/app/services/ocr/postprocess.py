"""
Post-OCR processing: entity extraction and text cleaning for Brazilian legal docs.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class LegalEntity:
    """Extracted legal entity from text."""

    text: str
    label: str  # CNJ_NUMBER, MONETARY_VALUE, LEGISLATION, ARTICLE_REF, SUMULA, etc.
    start: int
    end: int
    confidence: float


class LegalPostProcessor:
    """RegEx patterns and cleanup for Brazilian legal entities."""

    PATTERNS = {
        "cnj_number": re.compile(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}"),
        "monetary_value": re.compile(
            r"R\$\s?[\d.,]+(?:\s?(?:mil|milhões?|bilhões?))?"
        ),
        "legislation": re.compile(
            r"(?:Lei|Decreto|Resolução|Portaria|Instrução Normativa|"
            r"Medida Provisória|Emenda Constitucional)\s+"
            r"(?:n[ºo°]?\s*)?[\d.,/]+(?:\s*/\s*\d{4})?"
        ),
        "article_ref": re.compile(
            r"(?:Art(?:igo)?\.?\s*\d+(?:[°ºª])?"
            r"(?:\s*,\s*(?:§\s*\d+[°ºª]?|inciso\s+[IVXLCDM]+|alínea\s+[a-z]))*)"
        ),
        "sumula": re.compile(
            r"Súmula\s+(?:Vinculante\s+)?(?:n[ºo°]?\s*)?\d+"
            r"(?:\s+do\s+(?:STF|STJ|TST|TSE))?"
        ),
        "date_br": re.compile(
            r"\d{1,2}\s+de\s+(?:janeiro|fevereiro|março|abril|maio|junho|"
            r"julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4}"
        ),
        "cpf": re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}"),
        "cnpj": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
        "oab": re.compile(r"OAB[/\s]*[A-Z]{2}[\s/]*\d+"),
    }

    def extract_entities(self, text: str) -> List[LegalEntity]:
        """Extract structured legal entities using RegEx."""
        entities: List[LegalEntity] = []

        for label, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append(
                    LegalEntity(
                        text=match.group(),
                        label=label.upper(),
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,
                    )
                )

        return sorted(entities, key=lambda e: e.start)

    def clean_ocr_text(self, text: str) -> str:
        """Clean common OCR artifacts in Brazilian legal documents."""
        replacements = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            " .": ".",
            " ,": ",",
            " ;": ";",
            "\u00a0": " ",
            "  ": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Fix broken hyphenation (common in scanned docs)
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
