"""Tests for application-owned semantic score derivation."""

from src.role_coherence_monitor.scoring import derive_coherence_scores
from src.role_coherence_monitor.schemas import (
    CoherenceSignal,
    SignalSeverity,
    SignalSource,
    SignalType,
)


def make_signal(
    signal_type: SignalType,
    severity: SignalSeverity,
    ordinal: int = 1,
) -> CoherenceSignal:
    return CoherenceSignal(
        signal_id=f"semantic:turn-001:{signal_type.value}:{ordinal}",
        turn_id="turn-001",
        signal_type=signal_type,
        severity=severity,
        source=SignalSource.SEMANTIC,
        evidence_reference="agent_output",
        explanation="Synthetic scoring fixture.",
    )


def test_no_deviation_signals_produce_full_coherence_scores():
    scores = derive_coherence_scores(())

    assert scores.mission_alignment == 1.0
    assert scores.scope_adherence == 1.0
    assert scores.authority_adherence == 1.0
    assert scores.evidence_discipline == 1.0
    assert scores.behavioral_consistency == 1.0


def test_low_medium_high_map_to_application_owned_scores():
    scores = derive_coherence_scores(
        (
            make_signal(SignalType.MISSION_DRIFT, SignalSeverity.LOW),
            make_signal(SignalType.SCOPE_DRIFT, SignalSeverity.MEDIUM),
            make_signal(SignalType.AUTHORITY_EXPANSION, SignalSeverity.HIGH),
        )
    )

    assert scores.mission_alignment == 0.85
    assert scores.scope_adherence == 0.70
    assert scores.authority_adherence == 0.40
    assert scores.evidence_discipline == 1.0
    assert scores.behavioral_consistency == 1.0


def test_each_signal_type_changes_only_its_owned_dimension():
    scores = derive_coherence_scores(
        (
            make_signal(SignalType.EVIDENCE_DEGRADATION, SignalSeverity.HIGH),
        )
    )

    assert scores.mission_alignment == 1.0
    assert scores.scope_adherence == 1.0
    assert scores.authority_adherence == 1.0
    assert scores.evidence_discipline == 0.40
    assert scores.behavioral_consistency == 1.0


def test_multiple_signals_for_same_dimension_use_most_severe_score():
    scores = derive_coherence_scores(
        (
            make_signal(SignalType.SCOPE_DRIFT, SignalSeverity.LOW, 1),
            make_signal(SignalType.SCOPE_DRIFT, SignalSeverity.HIGH, 2),
            make_signal(SignalType.SCOPE_DRIFT, SignalSeverity.MEDIUM, 3),
        )
    )

    assert scores.scope_adherence == 0.40


def test_critical_mapping_is_fail_closed_for_non_live_callers():
    scores = derive_coherence_scores(
        (
            make_signal(SignalType.BEHAVIORAL_DRIFT, SignalSeverity.CRITICAL),
        )
    )

    assert scores.behavioral_consistency == 0.0


def test_recovery_signal_does_not_lower_any_dimension():
    scores = derive_coherence_scores(
        (
            make_signal(SignalType.RECOVERY, SignalSeverity.MEDIUM),
        )
    )

    assert scores.mission_alignment == 1.0
    assert scores.scope_adherence == 1.0
    assert scores.authority_adherence == 1.0
    assert scores.evidence_discipline == 1.0
    assert scores.behavioral_consistency == 1.0
