"""
Outcome predictor — XGBoost-based case outcome prediction.

Uses historical case data features to predict likely outcomes.
Criminal prediction is guarded per CNJ Res. 615/2025 Art. 23.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

CRIMINAL_AREA_LABELS = {"criminal", "penal", "execucao_penal"}

AREA_ENCODING = {
    "cível": 0, "civil": 0, "consumidor": 1, "trabalhista": 2,
    "tributário": 3, "tributario": 3, "administrativo": 4,
    "previdenciário": 5, "previdenciario": 5, "família": 6, "familia": 6,
    "ambiental": 7, "empresarial": 8, "imobiliário": 9, "imobiliario": 9,
}

TRIBUNAL_ENCODING = {
    "STF": 0, "STJ": 1, "TST": 2, "TSE": 3,
    "TJSP": 10, "TJRJ": 11, "TJMG": 12, "TJRS": 13, "TJPR": 14,
    "TJBA": 15, "TJSC": 16, "TJPE": 17, "TJCE": 18, "TJGO": 19,
    "TRF1": 20, "TRF2": 21, "TRF3": 22, "TRF4": 23, "TRF5": 24, "TRF6": 25,
}

OUTCOME_LABELS = ["procedente", "parcialmente_procedente", "improcedente", "extinto_sem_merito"]


class PredictionResult(BaseModel):
    outcome: str
    confidence: float
    probabilities: dict[str, float] = {}
    factors: list[dict[str, Any]] = []
    warning: Optional[str] = None


class OutcomePredictor:
    """XGBoost-based outcome predictor with feature engineering."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or "training/models/outcome_predictor.json"
        self._model = None
        self._feature_names = [
            "area_enc", "tribunal_enc", "valor_causa_log",
            "num_partes", "tipo_acao_enc", "judge_favorability",
            "court_procedencia_rate",
        ]

    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import xgboost as xgb
            model_file = Path(self.model_path)
            if model_file.exists():
                self._model = xgb.Booster()
                self._model.load_model(str(model_file))
                return True
        except ImportError:
            logger.debug("xgboost not installed, using heuristic fallback")
        except Exception as e:
            logger.warning("Failed to load XGBoost model: %s", e)
        return False

    def _extract_features(self, case_data: dict[str, Any]) -> list[float]:
        import math

        area = (case_data.get("area", "") or "").lower().strip()
        area_enc = float(AREA_ENCODING.get(area, -1))

        tribunal = (case_data.get("tribunal", case_data.get("court", "")) or "").upper()
        tribunal_enc = float(TRIBUNAL_ENCODING.get(tribunal, -1))

        valor = float(case_data.get("estimated_value", case_data.get("valor_causa", 0)) or 0)
        valor_log = math.log1p(valor)

        num_partes = float(case_data.get("num_partes", 2))
        tipo_enc = float(hash(case_data.get("tipo_acao", "")) % 50)

        judge_fav = float(case_data.get("judge_favorability", 50.0))
        court_rate = float(case_data.get("court_procedencia_rate", 50.0))

        return [area_enc, tribunal_enc, valor_log, num_partes, tipo_enc, judge_fav, court_rate]

    def predict(
        self,
        case_data: dict[str, Any],
        area: Optional[str] = None,
    ) -> PredictionResult:
        area_normalized = (area or case_data.get("area", "")).lower().strip()

        if area_normalized in CRIMINAL_AREA_LABELS:
            return PredictionResult(
                outcome="",
                confidence=0.0,
                factors=[],
                warning=(
                    "Predição em matéria criminal é desencorajada pela "
                    "CNJ Resolução 615/2025 Art. 23"
                ),
            )

        features = self._extract_features(case_data)

        if self._load_model():
            try:
                import xgboost as xgb
                import numpy as np

                dmat = xgb.DMatrix([features], feature_names=self._feature_names)
                probs = self._model.predict(dmat)[0]

                if isinstance(probs, (float, int)):
                    probs = [1 - probs, probs]

                probabilities = {}
                for i, label in enumerate(OUTCOME_LABELS[:len(probs)]):
                    probabilities[label] = round(float(probs[i]), 3)

                best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                outcome = OUTCOME_LABELS[best_idx] if best_idx < len(OUTCOME_LABELS) else "indeterminado"
                confidence = float(probs[best_idx])

                factors = self._explain_factors(features, case_data)

                return PredictionResult(
                    outcome=outcome,
                    confidence=round(confidence, 3),
                    probabilities=probabilities,
                    factors=factors,
                )
            except Exception as e:
                logger.warning("XGBoost prediction failed: %s", e)

        # Heuristic fallback when model not available
        return self._heuristic_predict(case_data, area_normalized, features)

    def _heuristic_predict(
        self, case_data: dict[str, Any], area: str, features: list[float]
    ) -> PredictionResult:
        """Rule-based fallback prediction."""
        judge_fav = features[5] if len(features) > 5 else 50.0
        court_rate = features[6] if len(features) > 6 else 50.0

        base_score = (judge_fav + court_rate) / 200.0

        probabilities = {
            "procedente": round(base_score * 0.5, 3),
            "parcialmente_procedente": round(base_score * 0.3, 3),
            "improcedente": round((1 - base_score) * 0.7, 3),
            "extinto_sem_merito": round((1 - base_score) * 0.1, 3),
        }

        best = max(probabilities, key=lambda k: probabilities[k])

        return PredictionResult(
            outcome=best,
            confidence=round(probabilities[best], 3),
            probabilities=probabilities,
            factors=[
                {"name": "Nota", "value": "Predição heurística (modelo XGBoost não treinado)"},
                {"name": "Taxa base do tribunal", "value": f"{court_rate:.0f}%"},
            ],
            warning="Predição baseada em heurística — modelo XGBoost ainda não treinado com dados reais.",
        )

    def _explain_factors(
        self, features: list[float], case_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Build human-readable explanation of prediction factors."""
        factors = []
        if features[0] >= 0:
            area_name = case_data.get("area", "desconhecida")
            factors.append({"name": "Área", "value": area_name, "importance": 0.25})
        if features[1] >= 0:
            tribunal = case_data.get("tribunal", case_data.get("court", ""))
            factors.append({"name": "Tribunal", "value": tribunal, "importance": 0.2})
        if features[2] > 0:
            factors.append({"name": "Valor da causa (log)", "value": round(features[2], 2), "importance": 0.15})
        if features[5] != 50.0:
            factors.append({"name": "Favorabilidade do juiz", "value": f"{features[5]:.1f}%", "importance": 0.3})
        return factors
