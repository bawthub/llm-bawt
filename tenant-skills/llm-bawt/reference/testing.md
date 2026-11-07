# Testing llm-bawt

Run the tests that exercise the change using the repo Make targets on the host from the host-visible task worktree:

```sh
make test-target NODES="tests/test_relevant.py"
```

From a bridge, run that command over SSH to the host, because the bridge has no pytest. Don't rebuild host dependencies inside a bridge. Use `make lint` when relevant. Add a meaningful regression for a bug fix when practical. Use existing deterministic test helpers rather than sleeps or tight wall-clock assertions.

The integrator runs one full `make gate REV=<batch-tip> AUTHOR=<bot>` after review. `make gate-receipt REV=<SHA>` on the host is the receipt for that exact published tip. Do not repeat a full suite for local iterations or at restart time when the exact receipt already exists. Keep a durable callback for detached checks that may outlive a turn.

`make test-sdk-probe` tests real CLI protocol behavior when that boundary changes. A dev VM via BawtHub `make vm-push VM=<name> SRC=<worktree>` is useful for an actual tenant-flow change, not a mandatory extra check for every edit. Test-runner details and suite registration live in repo `docs/TESTING.md`.
