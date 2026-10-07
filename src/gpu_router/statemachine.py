"""Job and attempt state machines as data (phase 1; real code, owner: group A).

This module is the single source of truth for which state changes are legal. The store
calls `check_transition()` before every write (invariant 3); CLAUDE.md's transition table
is generated from the same facts. It depends on nothing but the stdlib so the engine,
store, API and tests can all import it.

Job states
----------
queued             accepted; waiting to be placed (optionally until jobs.not_before)
routing            transient: the router is choosing a provider (no remote side effects)
awaiting_approval  the approval policy wants a human yes/no
provisioning       an attempt exists; submit in flight or remote run pending/starting
running            the remote run is executing the user's code
checkpointing      running, and the runner reported a checkpoint upload in progress
migrating          the previous attempt ended for an infrastructure reason; ensuring it is
                   terminal and choosing the next placement (resume from latest checkpoint)
cancelling         user asked to cancel; a live remote run is being stopped
done               remote run succeeded (outputs fetched or fetch failure explained)
failed             gave up: script error, nothing fits, attempts/wait budget exhausted, bug
cancelled          stopped by the user, or by someone outside gpu-router
denied             the approval request was denied or expired

Attempt states (one attempt = one placement of the job on one provider)
------------------------------------------------------------------------
submitting  row committed, adapter.submit not yet confirmed (crash window: resolve by key)
submitted   provider accepted it; remote_id known; remote pending
running     remote reported running
succeeded / failed / lost / cancelled   remote reached that end state
rejected    submit raised an AdapterError; nothing started remotely
abandoned   cannot tell whether a remote run exists (adapter lacks lookup_by_key); warned
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from gpu_router.errors import InvalidTransition


class JobState(StrEnum):
    QUEUED = "queued"
    ROUTING = "routing"
    AWAITING_APPROVAL = "awaiting_approval"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    MIGRATING = "migrating"
    CANCELLING = "cancelling"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DENIED = "denied"


class AttemptState(StrEnum):
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    LOST = "lost"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ABANDONED = "abandoned"


class Reason(StrEnum):
    """Machine codes for job_events.reason. Stable: `--json` consumers and tests match them.

    Add new members freely; never rename or repurpose one (API v1 is additive-only).
    """

    # lifecycle
    SUBMITTED = "submitted"  # (new) -> queued
    ROUTING_STARTED = "routing_started"  # queued -> routing
    PLACED = "placed"  # routing|migrating|awaiting_approval -> provisioning
    APPROVAL_REQUIRED = "approval_required"  # routing|migrating -> awaiting_approval
    APPROVED = "approved"  # note on awaiting_approval (driver then places)
    DENIED = "denied"  # awaiting_approval -> denied (user)
    APPROVAL_EXPIRED = "approval_expired"  # awaiting_approval -> denied (timeout)
    NO_CAPACITY = "no_capacity"  # routing|migrating|awaiting_approval -> queued (wait)
    NO_PROVIDER_FITS = "no_provider_fits"  # routing|migrating|awaiting_approval -> failed
    GAVE_UP = "gave_up"  # queued|routing|migrating -> failed (budget spent)
    STARTED = "started"  # provisioning -> running
    COMPLETED = "completed"  # provisioning|running|checkpointing|migrating -> done
    SCRIPT_FAILED = "script_failed"  # -> failed: user code exited non-zero
    # submit / provider trouble (provisioning -> queued unless noted)
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    QUOTA_EXHAUSTED = "quota_exhausted"  # also running -> migrating
    AUTH_REQUIRED = "auth_required"
    INVALID_FOR_PROVIDER = "invalid_for_provider"
    PROVIDER_PERMANENT = "provider_permanent"  # -> failed
    PROVISION_TIMEOUT = "provision_timeout"
    LOST_BEFORE_START = "lost_before_start"
    # migration
    SESSION_LOST = "session_lost"  # running|checkpointing -> migrating
    STATUS_LOST = "status_lost"  # running|checkpointing -> migrating (unreachable)
    HANDOFF = "handoff"  # running|checkpointing -> migrating (planned, phase 5)
    # running|checkpointing -> migrating: an agent job ran well past its declared hours;
    # it is placed again only after the user approves (D48)
    HOURS_EXCEEDED = "hours_exceeded"
    INTERACTIVE_LOST = "interactive_lost"  # running -> failed: interactive sessions never migrate
    # checkpoints
    CHECKPOINT_BEGIN = "checkpoint_begin"  # running -> checkpointing
    CHECKPOINT_END = "checkpoint_end"  # checkpointing -> running
    CHECKPOINT_STALLED = "checkpoint_stalled"  # checkpointing -> running (no end line in time)
    # cancellation
    USER_CANCEL = "user_cancel"  # -> cancelling | cancelled
    CANCELLED = "cancelled"  # cancelling -> cancelled (confirmed)
    CANCEL_UNCONFIRMED = "cancel_unconfirmed"  # cancelling -> cancelled (provider never confirmed)
    REMOTE_CANCELLED = "remote_cancelled"  # stopped outside gpu-router -> cancelled
    # notes (kind='note', no state change)
    SUBMIT_CONFIRMED = "submit_confirmed"
    RECOVERED = "recovered"
    STATUS_UNREACHABLE = "status_unreachable"
    STATUS_RECOVERED = "status_recovered"
    FETCHED = "fetched"
    FETCH_FAILED = "fetch_failed"
    PROVIDER_EXCLUDED = "provider_excluded"
    RETRY_SCHEDULED = "retry_scheduled"
    SUBMIT_AMBIGUOUS = "submit_ambiguous"  # submit timed out / Unavailable: resolving by key
    ATTEMPT_ABANDONED = "attempt_abandoned"  # could not tell if a remote run exists; warned
    ORPHAN_CANCELLED = "orphan_cancelled"  # a run found by an old attempt key was stopped
    FETCH_REQUESTED = "fetch_requested"
    OUTPUTS_KEPT = "outputs_kept"  # cancelled, but the remote had already succeeded
    INTERNAL_ERROR = "internal_error"  # note, or -> failed at the limit (not from cancelling)
    # checkpoint storage + data movement (phase 5, gpu_router/checkpoint; notes)
    HANDOFF_REQUESTED = "handoff_requested"  # asked the runner to checkpoint before a cap
    HANDOFF_SKIPPED = "handoff_skipped"  # the runner did not answer in time; runs to the end
    CHECKPOINT_FOUND = "checkpoint_found"  # storage had a newer checkpoint than the logs said
    CHECKPOINT_COPIED = "checkpoint_copied"  # moved between local and hf storage for an attempt
    STORAGE_UNAVAILABLE = "storage_unavailable"  # this attempt's checkpoints cannot move
    DATA_UPLOADED = "data_uploaded"  # a dataset was uploaded to storage (once per content)
    DATA_REUSED = "data_reused"  # a dataset upload was found in the data cache
    # a dataset is being uploaded to the provider's own store (Kaggle datasets, no
    # checkpoint storage needed; 2026-10-04): the job stays provisioning until it is there
    DATA_UPLOADING = "data_uploading"
    # the runner's nvidia-smi saw another GPU than the attempt was placed on (D56; note)
    GPU_MISMATCH = "gpu_mismatch"


TERMINAL_STATES: frozenset[JobState] = frozenset(
    {JobState.DONE, JobState.FAILED, JobState.CANCELLED, JobState.DENIED}
)

#: States in which the job may have a live remote run (an attempt that is not terminal).
REMOTE_STATES: frozenset[JobState] = frozenset(
    {
        JobState.PROVISIONING,
        JobState.RUNNING,
        JobState.CHECKPOINTING,
        JobState.MIGRATING,
        JobState.CANCELLING,
    }
)

_J = JobState

#: from-state -> legal to-states. Anything else raises InvalidTransition.
TRANSITIONS: Mapping[JobState, frozenset[JobState]] = MappingProxyType(
    {
        _J.QUEUED: frozenset({_J.ROUTING, _J.FAILED, _J.CANCELLED}),
        _J.ROUTING: frozenset(
            {_J.QUEUED, _J.AWAITING_APPROVAL, _J.PROVISIONING, _J.FAILED, _J.CANCELLED}
        ),
        _J.AWAITING_APPROVAL: frozenset(
            {_J.PROVISIONING, _J.QUEUED, _J.FAILED, _J.DENIED, _J.CANCELLED}
        ),
        _J.PROVISIONING: frozenset(
            {_J.RUNNING, _J.QUEUED, _J.DONE, _J.FAILED, _J.CANCELLING, _J.CANCELLED}
        ),
        _J.RUNNING: frozenset(
            {_J.CHECKPOINTING, _J.DONE, _J.FAILED, _J.MIGRATING, _J.CANCELLING, _J.CANCELLED}
        ),
        _J.CHECKPOINTING: frozenset(
            {_J.RUNNING, _J.DONE, _J.FAILED, _J.MIGRATING, _J.CANCELLING, _J.CANCELLED}
        ),
        _J.MIGRATING: frozenset(
            {
                _J.PROVISIONING,
                _J.QUEUED,
                _J.AWAITING_APPROVAL,
                _J.DONE,
                _J.FAILED,
                _J.CANCELLING,
                _J.CANCELLED,
            }
        ),
        _J.CANCELLING: frozenset({_J.CANCELLED}),
        _J.DONE: frozenset(),
        _J.FAILED: frozenset(),
        _J.CANCELLED: frozenset(),
        _J.DENIED: frozenset(),
    }
)

INITIAL_STATE = JobState.QUEUED


def is_terminal(state: JobState) -> bool:
    return state in TERMINAL_STATES


def can_transition(from_state: JobState, to_state: JobState) -> bool:
    return to_state in TRANSITIONS[from_state]


def check_transition(from_state: JobState, to_state: JobState) -> None:
    """Raise InvalidTransition unless from_state -> to_state is in TRANSITIONS."""
    if not can_transition(from_state, to_state):
        hint = "the job already finished" if is_terminal(from_state) else None
        raise InvalidTransition(str(from_state), str(to_state), hint=hint)


def cancel_target(state: JobState, *, has_live_attempt: bool) -> JobState | None:
    """Where a user cancel moves a job, or None if the job is already terminal.

    A job with a live attempt (submitting/submitted/running) goes to `cancelling` so the
    engine can stop the remote run; everything else goes straight to `cancelled`.
    `cancelling` itself returns None (cancel is idempotent; nothing more to do).
    """
    if is_terminal(state) or state is JobState.CANCELLING:
        return None
    if has_live_attempt and state in REMOTE_STATES:
        return JobState.CANCELLING
    return JobState.CANCELLED


# --------------------------------------------------------------------------- attempts

_A = AttemptState

LIVE_ATTEMPT_STATES: frozenset[AttemptState] = frozenset({_A.SUBMITTING, _A.SUBMITTED, _A.RUNNING})
TERMINAL_ATTEMPT_STATES: frozenset[AttemptState] = frozenset(AttemptState) - LIVE_ATTEMPT_STATES

ATTEMPT_TRANSITIONS: Mapping[AttemptState, frozenset[AttemptState]] = MappingProxyType(
    {
        _A.SUBMITTING: frozenset(
            {
                _A.SUBMITTED,
                _A.RUNNING,
                _A.SUCCEEDED,
                _A.FAILED,
                _A.LOST,
                _A.CANCELLED,
                _A.REJECTED,
                _A.ABANDONED,
            }
        ),
        _A.SUBMITTED: frozenset(
            {_A.RUNNING, _A.SUCCEEDED, _A.FAILED, _A.LOST, _A.CANCELLED, _A.ABANDONED}
        ),
        _A.RUNNING: frozenset({_A.SUCCEEDED, _A.FAILED, _A.LOST, _A.CANCELLED, _A.ABANDONED}),
        **{s: frozenset() for s in TERMINAL_ATTEMPT_STATES},
    }
)


def check_attempt_transition(from_state: AttemptState, to_state: AttemptState) -> None:
    if to_state not in ATTEMPT_TRANSITIONS[from_state]:
        raise InvalidTransition(f"attempt {from_state}", f"attempt {to_state}")


# --------------------------------------------------------------------------- crash recovery


class RecoveryAction(StrEnum):
    """What `Supervisor.start()` does for a job found in each state after a restart."""

    NONE = "none"  # terminal: nothing
    RESUME = "resume"  # just restart the driver (honours not_before)
    REROUTE = "reroute"  # routing had no side effects: re-run routing
    WAIT = "wait"  # keep waiting for approval (timeout from the transition ts)
    RESOLVE_ATTEMPT = "resolve_attempt"  # submitting w/o remote_id: status(key) then decide
    REATTACH = "reattach"  # poll the remote by remote_id; resume log capture
    REMIGRATE = "remigrate"  # re-run the (idempotent) migration steps
    RECANCEL = "recancel"  # re-issue cancel (idempotent), confirm, -> cancelled


RECOVERY: Mapping[JobState, RecoveryAction] = MappingProxyType(
    {
        _J.QUEUED: RecoveryAction.RESUME,
        _J.ROUTING: RecoveryAction.REROUTE,
        _J.AWAITING_APPROVAL: RecoveryAction.WAIT,
        _J.PROVISIONING: RecoveryAction.RESOLVE_ATTEMPT,
        _J.RUNNING: RecoveryAction.REATTACH,
        _J.CHECKPOINTING: RecoveryAction.REATTACH,
        _J.MIGRATING: RecoveryAction.REMIGRATE,
        _J.CANCELLING: RecoveryAction.RECANCEL,
        _J.DONE: RecoveryAction.NONE,
        _J.FAILED: RecoveryAction.NONE,
        _J.CANCELLED: RecoveryAction.NONE,
        _J.DENIED: RecoveryAction.NONE,
    }
)
