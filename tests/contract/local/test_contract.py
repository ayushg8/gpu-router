"""Every shared contract test (tests/contract/test_*.py), re-collected here so it runs with
this directory's `target`: LocalAdapter on this Mac (always, `env: system`) and with its
default uv venv (when GPU_ROUTER_REAL_PROVIDERS lists local). Extend the shared modules, not
this list; a new shared test only needs its name added below."""

from __future__ import annotations

from tests.contract.test_errors import (  # noqa: F401
    test_errors_carry_provider_and_message,
    test_failed_run_has_no_outputs_or_raises_not_found,
    test_fetch_of_unknown_run_raises_not_found,
    test_logs_of_unknown_run_raise_not_found,
    test_lookup_of_unknown_key_is_none,
    test_quota_exhausted_is_definitive_and_says_when,
    test_unavailable_submit_is_ambiguous_and_resolvable,
)
from tests.contract.test_fetch_cancel import (  # noqa: F401
    test_cancel_is_idempotent,
    test_cancel_stops_run,
    test_fetch_writes_outputs_and_is_rerunnable,
)
from tests.contract.test_logs import (  # noqa: F401
    test_cursor_resume_never_repeats_lines,
    test_fake_emits_metric_and_checkpoint_lines,
    test_logs_after_eof_are_empty,
)
from tests.contract.test_quota_health import (  # noqa: F401
    test_capabilities_are_consistent,
    test_healthcheck_ok_or_explains,
    test_quota_snapshot_shape,
)
from tests.contract.test_status import (  # noqa: F401
    test_nonzero_exit_is_failed,
    test_run_reaches_succeeded_with_exit_code_zero,
    test_session_death_is_lost,
    test_status_is_side_effect_free,
    test_unknown_remote_id_raises_not_found,
)
from tests.contract.test_submit import (  # noqa: F401
    test_definitive_submit_errors_create_nothing,
    test_lookup_by_key_finds_submitted_run,
    test_new_attempt_key_gets_new_run,
    test_submit_is_idempotent_per_attempt_key,
    test_submit_returns_remote_ref,
)
