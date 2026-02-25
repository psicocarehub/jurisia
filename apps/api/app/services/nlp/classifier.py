"""
Document classification for Brazilian legal documents.
Uses trained TF-IDF + LogReg model, with heuristic fallback.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

LABELS = [
    "peticao_inicial", "contestacao", "sentenca", "acordao",
    "decisao_monocratica", "recurso_apelacao", "recurso_agravo",
    "certidao", "procuracao", "laudo_parecer", "despacho", "outros",
]

_MODEL_DIR = Path(__file__).resolve().parents[5] / "training" / "models" / "doc_classifier"


class DocumentClassifier:
    """Classify legal document type. Uses trained model when available."""

    LABELS = LABELS

    def __init__(self, model_dir: str | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        self._pipeline = None
        self._vectorizer = None
        self._loaded = False
        self._labels = LABELS

    def _load_trained(self) -> bool:
        if self._loaded:
            return self._pipeline is not None

        self._loaded = True
        clf_path = self._model_dir / "classifier.joblib"
        vec_path = self._model_dir / "vectorizer.joblib"
        config_path = self._model_dir / "config.json"

        if clf_path.exists() and vec_path.exists():
            try:
                import joblib
                self._pipeline = joblib.load(clf_path)
                self._vectorizer = joblib.load(vec_path)
                if config_path.exists():
                    cfg = json.loads(config_path.read_text())
                    self._labels = cfg.get("labels", LABELS)
                logger.info("Loaded trained document classifier from %s (acc=%.1f%%)",
                            self._model_dir, cfg.get("accuracy", 0) * 100 if config_path.exists() else 0)
                return True
            except Exception as e:
                logger.warning("Failed to load trained classifier: %s", e)

        return False

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify document type using trained model or heuristic fallback."""
        if self._load_trained():
            try:
                snippet = text[:1500]
                X = self._vectorizer.transform([snippet])
                label_idx = self._pipeline.predict(X)[0]
                probs = self._pipeline.predict_proba(X)[0]
                confidence = float(probs.max())
                label = self._labels[label_idx] if label_idx < len(self._labels) else "outros"
                return label, confidence
            except Exception as e:
                logger.warning("Trained classifier prediction failed: %s", e)

        return self._heuristic_classify(text)

    def _heuristic_classify(self, text: str) -> Tuple[str, float]:
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
