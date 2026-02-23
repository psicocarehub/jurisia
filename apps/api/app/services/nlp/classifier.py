"""
Document classification for Brazilian legal documents.
16 labels. Stub that works without GPU/transformers.
"""

from typing import Tuple

# 16 document labels (BLUEPRINT)
LABELS = [
    "peticao_inicial",
    "contestacao",
    "replica",
    "sentenca",
    "acordao",
    "decisao_monocratica",
    "recurso_apelacao",
    "recurso_agravo",
    "recurso_extraordinario",
    "recurso_especial",
    "certidao",
    "procuracao",
    "comprovante",
    "laudo_parecer",
    "despacho",
    "outros",
]


class DocumentClassifier:
    """Classify legal document type. Stub: uses heuristics without transformers."""

    LABELS = LABELS

    def __init__(self, model_name: str = "pfreitag/roberta-legal-pt-classifier") -> None:
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self) -> bool:
        """Lazy load transformers pipeline. Returns False if not available."""
        if self._pipeline is not None:
            return True
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            return True
        except Exception:
            return False

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify document type. Uses first ~1000 tokens. Stub when no GPU."""
        if self._load_pipeline():
            inputs = text[:4000]
            try:
                result = self._pipeline(inputs, truncation=True, max_length=512)
                if result:
                    pred = result[0]
                    label = pred["label"].lower().replace(" ", "_")
                    if label not in self.LABELS:
                        label = "outros"
                    return label, float(pred.get("score", 0.8))
            except Exception:
                pass

        # Stub fallback: keyword-based heuristic
        return self._heuristic_classify(text)

    def _heuristic_classify(self, text: str) -> Tuple[str, float]:
        """Heuristic classification when transformers unavailable."""
        text_lower = text[:2000].lower()

        if "petição inicial" in text_lower or "peticao inicial" in text_lower:
            return "peticao_inicial", 0.7
        if "contestação" in text_lower or "contestacao" in text_lower:
            return "contestacao", 0.7
        if "sentença" in text_lower or "sentenca" in text_lower:
            return "sentenca", 0.7
        if "acórdão" in text_lower or "acordao" in text_lower:
            return "acordao", 0.7
        if "ementa" in text_lower and "relatório" in text_lower:
            return "acordao", 0.6
        if "recurso" in text_lower and "apelação" in text_lower:
            return "recurso_apelacao", 0.6
        if "certidão" in text_lower or "certidao" in text_lower:
            return "certidao", 0.7
        if "procuração" in text_lower or "procuracao" in text_lower:
            return "procuracao", 0.7

        return "outros", 0.5
