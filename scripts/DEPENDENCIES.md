# Source-install dependency contract

The initial complete installation target is **macOS arm64, Python 3.12.13**.
The Python/Node/pnpm versions come from `runtime-versions.conf`; the single
`requirements-macos-arm64.lock` freezes Python runtime, optional extras, test
tools, and the project's build backend. `pyproject.toml` remains the dependency
intent; the lock is its concrete installation closure, not a second runtime.

## Install from a clean checkout

Run at the repository root:

```sh
./scripts/bootstrap-local-runtime.sh
.venv/bin/python scripts/verify-runtime-independence.py
```

Bootstrap verifies the downloaded standalone Python/Node archives, installs
third-party wheels using the lock with `--require-hashes --only-binary=:all:`.
The sole source-only exception is OmegaConf's `antlr4-python3-runtime`: bootstrap
first installs setuptools from the same hash lock, then builds the hash-verified
antlr source with `--no-build-isolation`. No unpinned build environment is created.
It then installs this source checkout without resolving dependencies, runs
`pip check`, and installs the frontend with
the frozen pnpm lock. Runtime, pip, Corepack, and pnpm caches stay in `.runtime`.
The shell needs `curl`, `tar`, and `shasum` or `sha256sum`.

Do not copy an old `.venv`, `.runtime`, `node_modules`, or editable installation
to another checkout. Re-run bootstrap at the new location. The verifier checks
interpreter ownership and paths, not application readiness or model quality.

`--runtime-only` is available for the other standalone platforms listed in the
bootstrap script. It does **not** install dependencies, and is not a claim of a
complete, tested Linux/Windows/Intel Mac application installation. A macOS arm64
dependency lock must not silently be used as their lock.

## Explicit retrieval model preparation

Model weights are not in Git, the Python lock, or bootstrap. Check availability:

```sh
.venv/bin/python scripts/prepare-local-models.py check
```

This checks the canonical `model_assets` revisions/manifests locally, without
loading weights, starting a GPU, reading application state, or contacting an
LLM provider. Missing packages/assets produce structured `not_ready` and exit 1.
Only this explicit command downloads public Hub model assets:

```sh
.venv/bin/python scripts/prepare-local-models.py download
```

The download uses the same full commit SHAs as inference, without a Hub token;
it selects native weight/tokenizer/projection files rather than ONNX exports or
example images. Plan roughly 4–5 GB for the two model weight sets. It does not
download Docling layout/OCR assets. A successful structural check does not prove
an inference pass: after download, use the existing real offline smoke:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m personagraph.retrieval.operations.real_smoke
```

Hugging Face's normal `HF_HOME` / `HF_HUB_CACHE` configuration owns the cache.
If selecting a custom cache, set the same variables before both preparation
and GUI/runner startup; no symlink to a developer's cache is required. Never
package a private home directory or tokens with model assets. Weight licensing
and redistribution terms remain separate from the project's source license.

Docling remains an optional experimental reader. Installing `documents-layout`
does not supply its model resources; when its frozen local recipe cannot be
built, readers use native processing. To make the lightweight path explicit,
set `PERSONAGRAPH_DOCUMENT_ENGINE=native`. A Docling-specific reproduction also
needs matching package versions, layout/table commits, OCR backend, and recipe;
this retrieval setup script does not claim to prepare those resources.

## Update the one Python lock (maintainers)

Use macOS arm64 with the pinned project Python. Run standard `pip-tools==7.5.3`
with `pip==25.0.1` in a separate temporary tools virtualenv, not the runtime
environment. From the repository root, compile all extras and build dependencies:

```sh
pip-compile --verbose --no-config --resolver=backtracking --all-extras --all-build-deps \
  --strip-extras --allow-unsafe --generate-hashes --no-header --no-annotate \
  --no-emit-index-url --no-emit-trusted-host --index-url https://pypi.org/simple \
  --pip-args='--only-binary=:all: --no-binary=antlr4-python3-runtime --timeout=20 --retries=2' \
  --output-file requirements-macos-arm64.lock pyproject.toml
```

Do not inherit a private package index or credentials when generating a public
lock. Retain current pins by default; use explicit `--upgrade-package` only for
intentional updates. The initial lock is seeded with versions already used by
the project, not with a copied virtualenv. Unrelated packages are not included.
Review dependency changes and rerun a clean install, `pip check`, runtime path
verification, and the targeted installation tests before accepting a new lock.

```sh
.venv/bin/python -m pytest -q tests/scripts/test_source_installation.py
```

Source installation, model availability, real local retrieval, frontend launch,
and paid API/benchmark execution are separate acceptance steps. Passing an
earlier step must not be reported as passing the later ones.

The maintainer versions above are intentional. In the inspected pip-tools 7.6.1,
the public PyPI JSON lookup resolved to `https://pypi.org/<package>/json`, missing
the `/pypi/` path. Hash generation then downloaded unrelated platform wheels
instead of using release metadata. The unmodified 7.5.3 / pip 25.0.1 combination
uses `PackageIndex.pypi_url`; no project-specific resolver or third-party patch
is required. Do not upgrade this toolchain without rechecking metadata lookup.
