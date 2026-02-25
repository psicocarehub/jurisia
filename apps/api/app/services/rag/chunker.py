"""
Hierarchical chunking for Brazilian legal documents.
Uses ML section classifier when available, regex patterns as fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

SECTION_MODEL_PATH = Path(__file__).resolve().parents[5] / "training" / "models" / "section_classifier" / "section_classifier.joblib"
SECTION_CONFIG_PATH = SECTION_MODEL_PATH.parent / "config.json"


@dataclass
class Chunk:
    """A single chunk of text."""

    content: str
    content_type: str  # ementa, relatorio, fundamentacao, dispositivo, artigo, paragrafo, cabecalho, generic
    metadata: dict
    token_count: int


class LegalChunker:
    """Chunking for Brazilian legal documents (decisions, legislation, generic)."""

    SECTION_PATTERNS = {
        "ementa": r"(?i)(EMENTA|E\s*M\s*E\s*N\s*T\s*A)[:\s]",
        "relatorio": r"(?i)(RELATÓRIO|R\s*E\s*L\s*A\s*T\s*Ó\s*R\s*I\s*O)[:\s]",
        "fundamentacao": r"(?i)(FUNDAMENTAÇÃO|VOTO|DO MÉRITO|FUNDAMENTAÇÃO JURÍDICA)",
        "dispositivo": r"(?i)(DISPOSITIVO|DECISÃO|CONCLUSÃO|ISTO POSTO|ANTE O EXPOSTO)",
    }

    _section_clf = None
    _section_loaded = False

    @classmethod
    def _load_section_classifier(cls):
        if cls._section_loaded:
            return cls._section_clf is not None
        cls._section_loaded = True
        if SECTION_MODEL_PATH.exists():
            try:
                import joblib
                cls._section_clf = joblib.load(SECTION_MODEL_PATH)
                logger.info("Section classifier loaded")
                return True
            except Exception as e:
                logger.warning("Failed to load section classifier: %s", e)
        return False

    def _classify_section(self, text: str) -> str:
        """Classify text section using ML model."""
        if self._load_section_classifier():
            try:
                import json
                labels = ["ementa", "relatorio", "fundamentacao", "dispositivo", "cabecalho", "generic"]
                if SECTION_CONFIG_PATH.exists():
                    with open(SECTION_CONFIG_PATH) as f:
                        labels = json.load(f).get("labels", labels)
                pred = self._section_clf.predict([text[:500]])[0]
                return labels[pred] if isinstance(pred, int) else str(pred)
            except Exception as e:
                logger.warning("Section classifier prediction failed: %s", e)
        return "generic"

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, text: str, doc_type: str = "generic") -> List[Chunk]:
        """Dispatch to appropriate chunking strategy."""
        if doc_type in ("acordao", "sentenca", "decisao"):
            return self._chunk_judicial_decision(text)
        elif doc_type in ("lei", "decreto", "resolucao"):
            return self._chunk_legislation(text)
        else:
            return self._chunk_generic(text)

    def _chunk_judicial_decision(self, text: str) -> List[Chunk]:
        """Split judicial decisions by sections using regex, enhanced by ML classifier."""
        chunks: List[Chunk] = []
        sections = self._split_sections(text)

        for section_type, section_text in sections:
            if not section_text.strip():
                continue

            if section_type == "generic" and self._load_section_classifier():
                section_type = self._classify_section(section_text)

            if section_type == "ementa":
                chunks.append(
                    Chunk(
                        content=section_text.strip(),
                        content_type="ementa",
                        metadata={"section": "ementa"},
                        token_count=self._estimate_tokens(section_text),
                    )
                )
                continue

            paragraphs = self._split_paragraphs(section_text)
            section_chunks = self._merge_paragraphs_into_chunks(
                paragraphs, section_type
            )
            chunks.extend(section_chunks)

        return chunks

    def _chunk_legislation(self, text: str) -> List[Chunk]:
        """Split legislation by Art. pattern."""
        chunks: List[Chunk] = []
        articles = re.split(r"(?=Art\.\s*\d+)", text)

        for article in articles:
            if not article.strip():
                continue
            if self._estimate_tokens(article) <= self.max_chunk_tokens:
                chunks.append(
                    Chunk(
                        content=article.strip(),
                        content_type="artigo",
                        metadata=self._extract_article_number(article),
                        token_count=self._estimate_tokens(article),
                    )
                )
            else:
                sub_chunks = self._chunk_generic(article)
                for sc in sub_chunks:
                    sc.content_type = "artigo"
                chunks.extend(sub_chunks)

        return chunks

    def _chunk_generic(self, text: str) -> List[Chunk]:
        """Recursive paragraph splitting with overlap."""
        paragraphs = self._split_paragraphs(text)
        return self._merge_paragraphs_into_chunks(paragraphs, "generic")

    def _split_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split judicial decision into named sections."""
        positions: List[Tuple[int, str]] = []
        for section_type, pattern in self.SECTION_PATTERNS.items():
            for match in re.finditer(pattern, text):
                positions.append((match.start(), section_type))

        if not positions:
            return [("generic", text)]

        positions.sort(key=lambda x: x[0])
        sections: List[Tuple[str, str]] = []
        for i, (pos, stype) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            sections.append((stype, text[pos:end]))

        return sections

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_paragraphs_into_chunks(
        self, paragraphs: List[str], content_type: str
    ) -> List[Chunk]:
        """Merge paragraphs into chunks respecting max_tokens and overlap."""
        chunks: List[Chunk] = []
        current: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            if (
                current_tokens + para_tokens > self.max_chunk_tokens
                and current
            ):
                chunk_text = "\n\n".join(current)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        content_type=content_type,
                        metadata={},
                        token_count=current_tokens,
                    )
                )
                if self.overlap_tokens > 0 and current:
                    last = current[-1]
                    current = [last]
                    current_tokens = self._estimate_tokens(last)
                else:
                    current = []
                    current_tokens = 0

            current.append(para)
            current_tokens += para_tokens

        if current:
            chunks.append(
                Chunk(
                    content="\n\n".join(current),
                    content_type=content_type,
                    metadata={},
                    token_count=current_tokens,
                )
            )

        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (Portuguese: ~1 token per 4 chars)."""
        return len(text) // 4

    @staticmethod
    def _extract_article_number(text: str) -> dict:
        """Extract article number from text."""
        match = re.search(r"Art\.\s*(\d+)", text)
        return {"article_number": match.group(1)} if match else {}
