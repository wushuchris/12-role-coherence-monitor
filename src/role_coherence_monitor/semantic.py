"""Model-agnostic semantic assessment boundary for role coherence.

Semantic assessors may interpret language and long-horizon context, but they do
not own deterministic contract enforcement or final control-plane state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    AssessmentMode,
    CoherenceSignal,
    InteractionTurn,
    RoleContract,
    SignalSource,
    TurnAssessment,
)


class SemanticAssessmentError(ValueError):
    """Raised when semantic evidence violates the assessor contract."""


class SemanticAssessmentResult(BaseModel):
    """Structured semantic evidence without control-plane authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: str = Field(min_length=1)
    mission_alignment: float = Field(ge=0.0, le=1.0)
    scope_adherence: float = Field(ge=0.0, le=1.0)
    authority_adherence: float = Field(ge=0.0, le=1.0)
    evidence_discipline: float = Field(ge=0.0, le=1.0)
    behavioral_consistency: float = Field(ge=0.0, le=1.0)
    signals: tuple[CoherenceSignal, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_semantic_signal_boundary(self) -> "SemanticAssessmentResult":
        wrong_turn = [
            signal.signal_id for signal in self.signals if signal.turn_id != self.turn_id
        ]
        if wrong_turn:
            raise ValueError(
                "Semantic signals must belong to the assessed turn: " f"{wrong_turn}"
            )

        non_semantic = [
            signal.signal_id
            for signal in self.signals
            if signal.source is not SignalSource.SEMANTIC
        ]
        if non_semantic:
            raise ValueError(
                "SemanticAssessmentResult may contain only semantic signals: "
                f"{non_semantic}"
            )

        return self


@runtime_checkable
class SemanticAssessor(Protocol):
    """Interface implemented by mock and future live semantic assessors."""

    def assess(
        self,
        *,
        contract: RoleContract,
        turn: InteractionTurn,
        history: Sequence[InteractionTurn] = (),
    ) -> SemanticAssessmentResult:
        """Return bounded semantic evidence for one turn."""


def to_turn_assessment(
    result: SemanticAssessmentResult,
    *,
    assessment_id: str | None = None,
    mode: AssessmentMode = AssessmentMode.SEMANTIC,
) -> TurnAssessment:
    """Convert semantic evidence into the existing per-turn assessment contract."""

    return TurnAssessment(
        assessment_id=assessment_id or f"assessment:{result.turn_id}",
        turn_id=result.turn_id,
        mode=mode,
        mission_alignment=result.mission_alignment,
        scope_adherence=result.scope_adherence,
        authority_adherence=result.authority_adherence,
        evidence_discipline=result.evidence_discipline,
        behavioral_consistency=result.behavioral_consistency,
        signal_ids=tuple(signal.signal_id for signal in result.signals),
        rationale=result.rationale,
    )


class MockSemanticAssessor:
    """Deterministic fixture-backed assessor used before live model integration.

    The mock performs no natural-language inference. Tests and synthetic scenarios
    explicitly provide expected semantic evidence keyed by stable turn_id.
    """

    def __init__(self, results_by_turn_id: Mapping[str, SemanticAssessmentResult]):
        self._results_by_turn_id = dict(results_by_turn_id)

    def assess(
        self,
        *,
        contract: RoleContract,
        turn: InteractionTurn,
        history: Sequence[InteractionTurn] = (),
    ) -> SemanticAssessmentResult:
        del contract  # Interface parity; the mock does not reinterpret the contract.
        del history  # Fixture-backed behavior is intentionally deterministic.

        try:
            result = self._results_by_turn_id[turn.turn_id]
        except KeyError as exc:
            raise SemanticAssessmentError(
                f"No mock semantic result configured for turn_id '{turn.turn_id}'"
            ) from exc

        if result.turn_id != turn.turn_id:
            raise SemanticAssessmentError(
                "Configured semantic result does not match the requested turn_id"
            )

        return result
