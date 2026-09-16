"""Tests for deterministic role-contract enforcement."""

from src.role_coherence_monitor.deterministic import (
    check_turn_against_contract,
    has_hard_violation,
)
from src.role_coherence_monitor.schemas import (
    InteractionTurn,
    RoleContract,
    SignalSeverity,
    SignalSource,
    SignalType,
)


def make_role_contract() -> RoleContract:
    return RoleContract(
        role_id="compliance-reviewer-v1",
        role_name="Compliance Review Agent",
        mission="Review supplied evidence against policy and escalate exceptions.",
        allowed_responsibilities=(
            "Review supplied evidence",
            "Identify possible policy conflicts",
        ),
        prohibited_responsibilities=(
            "Approve transactions",
            "Redefine company policy",
        ),
        permitted_actions=(
            "request_clarification",
            "recommend_human_review",
        ),
        prohibited_actions=(
            "approve_transaction",
            "waive_policy",
        ),
        required_behaviors=("Cite supplied evidence",),
        escalation_conditions=("Consequential unresolved policy exception",),
        evidence_requirements=("Use only supplied or approved evidence",),
        version="1.0.0",
    )


def make_turn(**overrides) -> InteractionTurn:
    data = {
        "turn_id": "turn-001",
        "sequence_number": 1,
        "user_input": "Please review this exception.",
        "agent_output": "I will review the supplied evidence and policy.",
    }
    data.update(overrides)
    return InteractionTurn(**data)


def test_permitted_action_produces_no_signal():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(attempted_actions=("request_clarification",)),
    )

    assert signals == ()


def test_prohibited_action_produces_critical_hard_violation():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(attempted_actions=("approve_transaction",)),
    )

    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.PROHIBITED_ACTION
    assert signals[0].severity is SignalSeverity.CRITICAL
    assert signals[0].source is SignalSource.DETERMINISTIC
    assert has_hard_violation(signals) is True


def test_unknown_action_is_unauthorized_but_not_automatically_critical():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(attempted_actions=("send_external_email",)),
    )

    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.UNAUTHORIZED_ACTION
    assert signals[0].severity is SignalSeverity.HIGH
    assert has_hard_violation(signals) is False


def test_asserted_role_mismatch_is_flagged():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(provenance={"asserted_role_id": "transaction-approver-v2"}),
    )

    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.UNKNOWN_IDENTIFIER
    assert signals[0].evidence_reference == "provenance.asserted_role_id"


def test_asserted_contract_version_change_is_critical():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(provenance={"asserted_contract_version": "9.9.9"}),
    )

    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.CONTRACT_MUTATION
    assert signals[0].severity is SignalSeverity.CRITICAL
    assert has_hard_violation(signals) is True


def test_explicit_authority_and_contract_mutation_flags_are_critical():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(
            provenance={
                "attempted_contract_mutation": "true",
                "self_assigned_authority": "true",
            }
        ),
    )

    assert [signal.signal_type for signal in signals] == [
        SignalType.CONTRACT_MUTATION,
        SignalType.AUTHORITY_EXPANSION,
    ]
    assert all(signal.severity is SignalSeverity.CRITICAL for signal in signals)


def test_required_escalation_missing_is_critical():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(
            provenance={
                "escalation_required": "true",
                "escalation_performed": "false",
            }
        ),
    )

    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.MISSING_ESCALATION
    assert signals[0].severity is SignalSeverity.CRITICAL


def test_required_evidence_missing_is_high_severity():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(
            provenance={
                "evidence_required": "true",
                "evidence_present": "false",
            }
        ),
    )

    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.MISSING_EVIDENCE
    assert signals[0].severity is SignalSeverity.HIGH


def test_malformed_boolean_marker_is_reported():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(provenance={"self_assigned_authority": "sometimes"}),
    )

    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.MALFORMED_RECORD
    assert signals[0].severity is SignalSeverity.HIGH


def test_deterministic_checker_does_not_infer_semantics_from_prose():
    signals = check_turn_against_contract(
        make_role_contract(),
        make_turn(
            agent_output=(
                "I have decided that I can approve this transaction and waive the policy."
            )
        ),
    )

    assert signals == ()


def test_signal_ids_are_stable_for_same_turn_and_structured_facts():
    contract = make_role_contract()
    turn = make_turn(
        attempted_actions=("approve_transaction", "send_external_email"),
        provenance={"self_assigned_authority": "true"},
    )

    first = check_turn_against_contract(contract, turn)
    second = check_turn_against_contract(contract, turn)

    assert [signal.signal_id for signal in first] == [
        signal.signal_id for signal in second
    ]
    assert len({signal.signal_id for signal in first}) == len(first)
