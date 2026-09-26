"""The job/attempt state machines: every (from, to) pair against CLAUDE.md's table."""

from __future__ import annotations

import itertools

import pytest

from gpu_router.errors import InvalidTransition
from gpu_router.statemachine import (
    ATTEMPT_TRANSITIONS,
    LIVE_ATTEMPT_STATES,
    RECOVERY,
    REMOTE_STATES,
    TERMINAL_ATTEMPT_STATES,
    TERMINAL_STATES,
    TRANSITIONS,
    AttemptState,
    JobState,
    RecoveryAction,
    can_transition,
    cancel_target,
    check_attempt_transition,
    check_transition,
    is_terminal,
)

J = JobState
A = AttemptState

# Written out independently of statemachine.py, from CLAUDE.md "Job state machine".
EXPECTED: dict[JobState, set[JobState]] = {
    J.QUEUED: {J.ROUTING, J.FAILED, J.CANCELLED},
    J.ROUTING: {J.QUEUED, J.AWAITING_APPROVAL, J.PROVISIONING, J.FAILED, J.CANCELLED},
    J.AWAITING_APPROVAL: {J.PROVISIONING, J.QUEUED, J.FAILED, J.DENIED, J.CANCELLED},
    J.PROVISIONING: {J.RUNNING, J.QUEUED, J.DONE, J.FAILED, J.CANCELLING, J.CANCELLED},
    J.RUNNING: {J.CHECKPOINTING, J.DONE, J.FAILED, J.MIGRATING, J.CANCELLING, J.CANCELLED},
    J.CHECKPOINTING: {J.RUNNING, J.DONE, J.FAILED, J.MIGRATING, J.CANCELLING, J.CANCELLED},
    J.MIGRATING: {
        J.PROVISIONING,
        J.QUEUED,
        J.AWAITING_APPROVAL,
        J.DONE,
        J.FAILED,
        J.CANCELLING,
        J.CANCELLED,
    },
    J.CANCELLING: {J.CANCELLED},
    J.DONE: set(),
    J.FAILED: set(),
    J.CANCELLED: set(),
    J.DENIED: set(),
}

EXPECTED_ATTEMPT: dict[AttemptState, set[AttemptState]] = {
    A.SUBMITTING: {
        A.SUBMITTED,
        A.RUNNING,
        A.SUCCEEDED,
        A.FAILED,
        A.LOST,
        A.CANCELLED,
        A.REJECTED,
        A.ABANDONED,
    },
    A.SUBMITTED: {A.RUNNING, A.SUCCEEDED, A.FAILED, A.LOST, A.CANCELLED, A.ABANDONED},
    A.RUNNING: {A.SUCCEEDED, A.FAILED, A.LOST, A.CANCELLED, A.ABANDONED},
    A.SUCCEEDED: set(),
    A.FAILED: set(),
    A.LOST: set(),
    A.CANCELLED: set(),
    A.REJECTED: set(),
    A.ABANDONED: set(),
}


def test_table_covers_every_state() -> None:
    assert set(TRANSITIONS) == set(JobState)
    assert set(ATTEMPT_TRANSITIONS) == set(AttemptState)


@pytest.mark.parametrize(
    ("src", "dst"), list(itertools.product(JobState, JobState)), ids=lambda s: str(s)
)
def test_every_job_pair(src: JobState, dst: JobState) -> None:
    legal = dst in EXPECTED[src]
    assert can_transition(src, dst) is legal
    if legal:
        check_transition(src, dst)
    else:
        with pytest.raises(InvalidTransition) as info:
            check_transition(src, dst)
        assert info.value.detail == {"from": str(src), "to": str(dst)}
        assert info.value.http_status == 409


@pytest.mark.parametrize(
    ("src", "dst"), list(itertools.product(AttemptState, AttemptState)), ids=lambda s: str(s)
)
def test_every_attempt_pair(src: AttemptState, dst: AttemptState) -> None:
    if dst in EXPECTED_ATTEMPT[src]:
        check_attempt_transition(src, dst)
    else:
        with pytest.raises(InvalidTransition):
            check_attempt_transition(src, dst)


def test_terminal_states_have_no_exits_and_hint() -> None:
    assert {J.DONE, J.FAILED, J.CANCELLED, J.DENIED} == TERMINAL_STATES
    for state in TERMINAL_STATES:
        assert is_terminal(state)
        assert not TRANSITIONS[state]
        with pytest.raises(InvalidTransition) as info:
            check_transition(state, J.QUEUED)
        assert info.value.hint == "the job already finished"
    with pytest.raises(InvalidTransition) as info:
        check_transition(J.QUEUED, J.DONE)
    assert info.value.hint is None


def test_every_non_terminal_state_can_reach_a_terminal_state() -> None:
    for start in JobState:
        seen = {start}
        frontier = [start]
        while frontier:
            for nxt in TRANSITIONS[frontier.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        assert seen & TERMINAL_STATES, start


def test_every_non_terminal_state_can_be_cancelled() -> None:
    for state in set(JobState) - TERMINAL_STATES:
        assert can_transition(state, J.CANCELLED), state


def test_attempt_state_partitions() -> None:
    assert {A.SUBMITTING, A.SUBMITTED, A.RUNNING} == LIVE_ATTEMPT_STATES
    assert set(AttemptState) == LIVE_ATTEMPT_STATES | TERMINAL_ATTEMPT_STATES
    assert not LIVE_ATTEMPT_STATES & TERMINAL_ATTEMPT_STATES


@pytest.mark.parametrize("state", list(JobState), ids=str)
@pytest.mark.parametrize("live", [True, False])
def test_cancel_target(state: JobState, live: bool) -> None:
    target = cancel_target(state, has_live_attempt=live)
    if is_terminal(state) or state is J.CANCELLING:
        assert target is None
    elif live and state in REMOTE_STATES:
        assert target is J.CANCELLING
    else:
        assert target is J.CANCELLED
    if target is not None:
        assert can_transition(state, target)


def test_recovery_covers_every_state() -> None:
    assert set(RECOVERY) == set(JobState)
    assert RECOVERY[J.QUEUED] is RecoveryAction.RESUME
    assert RECOVERY[J.ROUTING] is RecoveryAction.REROUTE
    assert RECOVERY[J.AWAITING_APPROVAL] is RecoveryAction.WAIT
    assert RECOVERY[J.PROVISIONING] is RecoveryAction.RESOLVE_ATTEMPT
    assert RECOVERY[J.RUNNING] is RecoveryAction.REATTACH
    assert RECOVERY[J.CHECKPOINTING] is RecoveryAction.REATTACH
    assert RECOVERY[J.MIGRATING] is RecoveryAction.REMIGRATE
    assert RECOVERY[J.CANCELLING] is RecoveryAction.RECANCEL
    for state in TERMINAL_STATES:
        assert RECOVERY[state] is RecoveryAction.NONE


def test_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        TRANSITIONS[J.DONE] = frozenset({J.QUEUED})  # type: ignore[index]
