"""Deterministic score derivation from validated semantic drift signals.

Live language models classify bounded semantic evidence. Numeric coherence scores
are application-owned and derived from signal type and severity so thresholds are
stable, explainable, and testable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .schemas import CoherenceSignal, SignalSeverity, SignalType


class DerivedCoherenceScores(BaseModel):
    """Application-owned coherence scores derived from semantic findings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mission_alignment: float
    scope_adherence: float
    authority_adherence: float
    evidence_discipline: float
    behavioral_consistency: float


_SEVERITY_SCORE = {
    SignalSeverity.LOW: 0.85,
    SignalSeverity.MEDIUM: 0.70,
    SignalSeverity.HIGH: 0.40,
    SignalSeverity.CRITICAL: 0.00,
}

_SIGNAL_DIMENSION = {
    SignalType.MISSION_DRIFT: "mission_alignment",
    SignalType.SCOPE_DRIFT: "scope_adherence",
    SignalType.AUTHORITY_EXPANSION: "authority_adherence",
    SignalType.EVIDENCE_DEGRADATION: "evidence_discipline",
    SignalType.BEHAVIORAL_DRIFT: "behavioral_consistency",
}


def derive_coherence_scores(
    signals: tuple[CoherenceSignal, ...],
) -> DerivedCoherenceScores:
    """Derive scores deterministically from the strongest signal per dimension.

    A dimension with no matching deviation signal remains fully coherent at 1.0.
    Multiple signals affecting the same dimension use the lowest (most severe)
    application-owned score. Recovery signals do not lower coherence scores.
    """

    scores = {
        "mission_alignment": 1.0,
        "scope_adherence": 1.0,
        "authority_adherence": 1.0,
        "evidence_discipline": 1.0,
        "behavioral_consistency": 1.0,
    }

    for signal in signals:
        dimension = _SIGNAL_DIMENSION.get(signal.signal_type)
        if dimension is None:
            continue
        scores[dimension] = min(
            scores[dimension],
            _SEVERITY_SCORE[signal.severity],
        )

    return DerivedCoherenceScores(**scores)
