# Testing llm-bawt

Run the tests that exercise the change using the repo Make targets on the host from the host-visible task worktree:

```sh
make test-target NODES="tests/test_relevant.py"
```

From a bridge, run that command over SSH to the host, because the bridge has no pytest; in a house, plain `pytest <named files>` works in place. Keep host runs to named files or nodes; broad subsets and project-wide typechecks go through the heavy queue (see houses.md). Neither the queue nor a house can run the full unit suite or gate. Don't rebuild host dependencies inside a bridge. Use `make lint` when relevant. Add a meaningful regression for a bug fix when practical. Use existing deterministic test helpers rather than sleeps or tight wall-clock assertions.

The integrator runs one full `make gate REV=<batch-tip> AUTHOR=<bot>` after review. If an owner genuinely needs a full pre-check first, the integrator runs it as one solo host gate on the reviewed tip. `make gate-receipt REV=<SHA>` on the host is the receipt for that exact published tip. Do not repeat a full suite for local iterations or at restart time when the exact receipt already exists. Keep a durable callback for detached checks that may outlive a turn.

`make test-sdk-probe` tests real CLI protocol behavior when that boundary changes. A dev VM via BawtHub `make vm-push VM=<name> SRC=<worktree>` is useful for an actual tenant-flow change, not a mandatory extra check for every edit. Test-runner details and suite registration live in repo `docs/TESTING.md`.
