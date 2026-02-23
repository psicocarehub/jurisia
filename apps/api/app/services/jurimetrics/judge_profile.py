"""
Jurimetrics: judge profiles and outcome prediction.
CNJ Res. 615/2025: criminal prediction discouraged.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class JudgeProfile(BaseModel):
    """Judge profile from aggregated data."""

    name: str
    court: str
    jurisdiction: str
    total_decisions: int = 0
    avg_decision_time_days: float = 0.0
    favorability: Dict[str, Dict[str, float]] = {}  # {area: {autor: %, réu: %}}
    top_citations: list = []
    decision_patterns: Dict[str, Any] = {}
    conciliation_rate: float = 0.0
    reform_rate: float = 0.0


class JurimetricsService:
    """Judge profiling and outcome prediction."""

    async def get_judge_profile(
        self, judge_name: str, court: Optional[str] = None
    ) -> Optional[JudgeProfile]:
        """Build judge profile from multiple sources."""
        # TODO: DB cache + external APIs (Data Lawyer, Turivius, JUDIT)
        return None

    async def predict_outcome(
        self,
        case_area: str,
        judge_name: str,
        case_features: dict,
    ) -> Dict[str, float]:
        """Predict case outcome probability. Criminal guard per CNJ Res. 615/2025."""
        if case_area.lower() in ("criminal", "penal"):
            raise ValueError(
                "Predição em matéria criminal é desencorajada pela "
                "CNJ Resolução 615/2025 Art. 23"
            )
        # TODO: XGBoost model
        return {}
