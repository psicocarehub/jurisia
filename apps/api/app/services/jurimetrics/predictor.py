"""
Outcome predictor — predição de resultado (XGBoost).

Predição de desfecho processual com guard para matéria criminal
(CNJ Res. 615/2025 Art. 23).
"""

from typing import Any, Optional

from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Resultado da predição."""

    outcome: str  # pro author, pro defendant, parcial, etc.
    confidence: float
    factors: list[dict[str, Any]] = []
    warning: Optional[str] = None


# Áreas onde predição é desencorajada (CNJ Res. 615/2025 Art. 23)
CRIMINAL_AREA_LABELS = {"criminal", "penal", "execucao_penal"}


class OutcomePredictor:
    """
    Preditor de resultado processual baseado em XGBoost (stub).

    IMPORTANTE: Modelos preditivos em matéria CRIMINAL são
    desencorajados pela CNJ Resolução 615/2025 Art. 23.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self._model = None

    def _load_model(self) -> bool:
        """Carrega modelo XGBoost (stub)."""
        if self._model is not None:
            return True
        # Stub: XGBoost.load_model()
        return False

    def _extract_features(self, case_data: dict[str, Any]) -> list[float]:
        """Feature engineering a partir dos dados do caso (stub)."""
        _ = case_data
        return []

    def predict(
        self,
        case_data: dict[str, Any],
        area: Optional[str] = None,
    ) -> PredictionResult:
        """
        Prediz resultado do caso.

        Args:
            case_data: Dados do caso (partes, assunto, tribunal, etc.)
            area: Área do direito. Se criminal, retorna aviso.

        Returns:
            PredictionResult com outcome e confidence.
            Em matéria criminal, inclui warning conforme CNJ Res. 615/2025.
        """
        area_normalized = (area or "").lower().strip()

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

        # Stub: XGBoost inference
        _ = case_data
        return PredictionResult(
            outcome="",
            confidence=0.0,
            factors=[],
        )
