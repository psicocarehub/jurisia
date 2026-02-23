"""
Legal NER: Transformer (pierreguillou/ner-bert-large-cased-pt-lenerbr) + RegEx.
"""

from typing import List

from app.services.ocr.postprocess import LegalPostProcessor, LegalEntity


class LegalNERService:
    """Combined NER: Transformer + RegEx with deduplication."""

    def __init__(self) -> None:
        self.regex_extractor = LegalPostProcessor()
        self._ner_pipeline = None

    def _load_ner(self) -> bool:
        """Lazy load transformers NER pipeline."""
        if self._ner_pipeline is not None:
            return True
        try:
            from transformers import pipeline

            self._ner_pipeline = pipeline(
                "ner",
                model="pierreguillou/ner-bert-large-cased-pt-lenerbr",
                aggregation_strategy="simple",
            )
            return True
        except Exception:
            return False

    def extract_entities(self, text: str) -> List[LegalEntity]:
        """Combined NER: Transformer + RegEx, deduplicated."""
        entities: List[LegalEntity] = []

        # 1. Transformer NER (chunks of ~2000 chars with overlap)
        if self._load_ner():
            for i in range(0, len(text), 1800):
                chunk = text[i : i + 2000]
                try:
                    ner_results = self._ner_pipeline(chunk)
                    for ent in ner_results:
                        entities.append(
                            LegalEntity(
                                text=ent.get("word", ""),
                                label=ent.get("entity_group", "O"),
                                start=ent.get("start", 0) + i,
                                end=ent.get("end", 0) + i,
                                confidence=float(ent.get("score", 0.8)),
                            )
                        )
                except Exception:
                    continue

        # 2. RegEx entities
        regex_entities = self.regex_extractor.extract_entities(text)
        entities.extend(regex_entities)

        # 3. Deduplicate overlapping entities (prefer higher confidence)
        entities = self._deduplicate(entities)

        return entities

    def _deduplicate(
        self, entities: List[LegalEntity]
    ) -> List[LegalEntity]:
        """Remove overlapping entities, preferring higher confidence."""
        if not entities:
            return []

        entities.sort(key=lambda e: (e.start, -e.confidence))
        result = [entities[0]]

        for ent in entities[1:]:
            last = result[-1]
            if ent.start >= last.end:
                result.append(ent)
            elif ent.confidence > last.confidence:
                result[-1] = ent

        return result
