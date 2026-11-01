# Finite offline runner assets (TASK-1865)

This is **local packaging evidence**, not an ops action, built runner image,
complete offline dependency set, toolchain attestation or SHIP gate. It does not
change the worker, catalog, policy, services, shared checkout or host receipts.
The existing 11,193/11,193 sandbox feasibility result stands; do not rerun a full
gate merely to validate this manifest.

## Fixed scope and provenance

`runner-assets/manifest.v1.json` is a content inventory for Linux x86-64 with:

- `o200k_base.tiktoken`, using the SHA256 declared by locked tiktoken 0.12.0.
  Its cache filename is the SHA1 of the complete upstream URL, **not** a payload
  integrity hash. Mount `tiktoken/` read-only and set `TIKTOKEN_CACHE_DIR` to it.
- A versioned public Debian `media-types_13.0.0_all.deb` and its exact
  `./etc/mime.types` member. The file was extracted without installing the package
  or executing maintainer scripts. Put the pinned MIME file at `/etc/mime.types`
  in a future private runner. This fixes the missing `image/webp` mapping; it is
  not a new application MIME policy.
- Ten exact source/lockfile inputs from llm-bawt
  `56d2f33808bdffd1eba294dc94e1a27615647c9f` and BawtHub
  `d9ca409d7dff0b6182dd637a0fbf35ed7191990d`, exported from committed bytes.
  Their inventory paths preserve source-relative paths under `sources/<repo>/`.

Tokenizer bytes came from the public upstream URL and matched the package's
expected digest. MIME bytes came from the public versioned Debian HTTPS URL;
this manifest pins their observed SHA256. **No Debian signed release-chain
verification or distribution attestation is claimed.** No payload is sourced
from a host tokenizer cache, pnpm store, installed `node_modules` or live DB.
URLs/origin notes are provenance documentation, not executable fetch commands.
Re-acquisition must match every size and SHA256; mirrors may disappear, but a
substitute with different bytes is refused.

The validator does not fetch, extract archives, run a tool, install dependencies,
open a socket, access Docker/PG, launch a runner or publish a receipt. Payloads
are kept outside git; this small committed manifest is their immutable identity.
A staging owner can acquire the public payloads and export the pinned git files
before entering an offline sandbox. Freeze that private staging before validation
and subsequent use; validation is not protection from concurrent replacement by
an untrusted writer. It hashes only inventoried files; do not mount arbitrary
extra staging contents into a runner as if they had been validated.

## Exact tool requirements versus toolchain proof

Preserved BawtHub CI requires Node **24.21.0**, uv **0.10.8**, Corepack **0.36.0**
and pnpm **10.32.1**. The packageManager field also supplies pnpm's SHA512
integrity string; the exact source file is inventoried. The prior PoC's uv
0.13.0 does **not** satisfy this contract.

Preserved llm-bawt CI declares Python **3.12**, not a patch/build, and
`setup-uv@v6`, which pins the GitHub action, **not** uv binary version 6. The
finite common requirement therefore accepts Python 3.12.x and uses BawtHub's
exact uv requirement. It must not claim that Python 3.12.13, 3.12.15 or any
other observed host version is an exact upstream CI pin.

A declared-version JSON comparison input has exactly these five keys. Example
only, **not an observation or attestation**:

```json
{"python":"3.12.13","node":"24.21.0","uv":"0.10.8","corepack":"0.36.0","pnpm":"10.32.1"}
```

The validator compares these strings; it does **not** prove binary identity,
platform, execution or authenticity. Actual future packaging must pin tool
artifacts and collect trusted version/binary evidence. Do not treat a successful
comparison with this example as proof that a toolchain is installed.

## Offline validation

Obtain the expected manifest SHA256 from the reviewed revision, **not** from the
same untrusted staged manifest. For this v1 file it is:

`93e60e3ad5dd6f2953630d15a8c7955552ce2feebff4c1e247efa1ec5aed1716`

Using a real private staging directory and a separately collected declaration:

```sh
python3 -m scripts.offline_runner_assets \
  --bundle /private/staged-assets \
  --manifest runner-assets/manifest.v1.json \
  --manifest-sha256 93e60e3ad5dd6f2953630d15a8c7955552ce2feebff4c1e247efa1ec5aed1716 \
  --declared-tool-versions /private/declared-tool-versions.json
```

Exit 0 means only `verified_finite_files_and_declared_versions`, with
`attestation`, `complete_offline_ci` and `gate_receipt` all **false**. Missing,
changed, nonregular or symlinked inventoried files; path traversal; duplicate
JSON keys/paths; unsupported schema; and version mismatches refuse (CLI exit 2).
Source/lock changes require a new reviewed manifest identity, not silent reuse
for a different SHA. The validator never writes to staging.

Focused regression target (host task worktree):

```sh
make test-target NODES=tests/test_offline_runner_assets.py
```

## Explicit remaining limits

This inventory does **not** contain tool binaries, Python wheels, pnpm package
store/native assets or a runner image. BawtHub's broader precommit path also
uses Python 3.14, pyright 1.1.409, Home Assistant 2026.10.2 and dynamic extras;
those environments are not closed by the root uv.lock. Corepack provisioning,
package build scripts, Prisma engines and other remote hook assets remain
outside this finite bundle. Only o200k_base is covered, not arbitrary encodings.

Socketless runner/sidecar isolation and existing design/security approvals are
unchanged. Durable typed gate jobs, dispatch/retry fences, receipts/callbacks,
image build/repin, publication and activation are not implemented or authorized
by this packaging slice. Normal host exact-SHA gates remain shipping authority.
