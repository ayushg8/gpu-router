A fake `lightning_sdk` for tests/unit/providers/lightning/test_driver.py. The driver runs in
a subprocess with this directory on PYTHONPATH; state comes from the JSON file named by
FAKE_LIGHTNING_STATE and every call is appended to FAKE_LIGHTNING_CALLS (JSON lines). Only
the names and signatures driver.py uses exist, copied from lightning-sdk 2026.9.18.post1
(job.py, teamspace.py, studio.py, machine.py, status.py, utils/resolve.py, filesystem.py,
lightning_cloud/rest_client.py, lightning_cloud/openapi/rest.py). Not a test package:
nothing here is collected by pytest.
