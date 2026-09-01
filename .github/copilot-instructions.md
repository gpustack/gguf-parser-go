# Copilot Code Review — GGUF Parser

GGUF Parser is a Go library and CLI that parses
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model files — the binary format used by
GGML-based executors such as llama.cpp — and estimates their memory usage and maximum tokens per second (TPS)
without downloading whole files. It reads the metadata of a local file, a remote URL (via chunked range reads),
or a model referenced from the Ollama library, HuggingFace, or ModelScope, and projects per-device RAM/VRAM usage
across full, zero, or partial GPU layer offloads, including multimodal projectors and StableDiffusion.Cpp-style
compound files. Estimates track llama.cpp's allocation behavior; deviation from actual usage is typically around
100MiB. Read `README.md` first for the estimation semantics a PR touches.

Layout: two Go modules — the root library module and the CLI module under `cmd/gguf-parser` (`urfave/cli/v2`,
dot-importing the root package). The root package (`gguf_parser`) holds the public API and core logic in
`file*.go` (parsing, metadata, architecture, tokenizer, remote/distro readers, memory/TPS estimation under
`file_estimate__*.go`), plus `ggml.go` and `scalar.go` for the GGML type system. Shared helpers live under
`util/` (`anyx`, `bytex`, `funcx`, `httpx`, `json`, `osx`, `ptr`, `signalx`, `slicex`, `stringx`). Code
generation is declared in `gen.go`: two `//go:generate` stages (`gen.stringer.go` with `-tags stringer`,
`gen.regression.go` with `-tags regression`) emit `zz_generated.*` files. The Makefile targets are `deps`,
`generate`, `lint`, `test`, `benchmark`, `build`/`package`, and `ci` (the default: deps → generate → lint →
test → build), each covering both modules; linting is governed by `.golangci.yaml` (gofumpt with extra rules,
godot, lll at 150 columns, importas with no unaliased imports, and more), with `goimports-reviser` and
`golangci-lint` downloaded into `.sbin/`.

When performing a code review, apply the rules below. Keep feedback specific and actionable; cite the file and
line.

## Out of scope — do not review

- Files matching `zz_generated*`, `gen.*` — generated code and its generators' build-tagged scaffolding.
- `.sbin/` (downloaded lint tooling), `.dist/` (build artifacts).
- Vendored or downloaded tooling of any kind.

## Hard invariants — flag as required changes

- PRs that edit types or enums feeding the generators (e.g. GGML types, quantizations, architectures) must
  include the regenerated `zz_generated.*` files (`make generate`). Never hand-edit `zz_generated.*` or `gen.*`
  files.
- Both Go modules must build and pass checks — `make ci` covers `deps`, `generate`, `lint`, `test`, and `build`
  for the root library and `cmd/gguf-parser` together; flag a change that leaves one module behind.
- Parsing must never panic on malformed input — return an error. The repo recently fixed several parse panics
  (unbounded declared array/string lengths, typeless projectors); flag any new trust in untrusted file data.
- Estimates must stay aligned with llama.cpp's allocation behavior; flag estimation changes without a cited
  upstream reference or test coverage.
- Commit messages follow Conventional Commits (`type: subject`, e.g. `fix: ...`, `refactor: ...`).

## Go conventions

- Favor clear code over cleverness; flag needless complexity or speculative abstraction.
- Errors must be handled explicitly; flag panics used for control flow.
- Keep interfaces small — accept interfaces, return concrete types.
- Use concise, meaningful names; multi-word Go files are snake_case (`file_metadata.go`), never
  flat-concatenated.
- Prefer short, single-purpose functions; favor composition and value semantics.
- Keep concurrency simple and minimal — the parser leans on mmap and chunked/ring-buffer reads, so flag shared
  mutable state that risks data races.
- Exported APIs need doc comments describing behavior, expectations, and constraints (godot: full sentences
  ending in a period).
- Use `any`, not `interface{}` (enforced by gofmt rewrite rules in `.golangci.yaml`); respect the import order
  std/general/company/project applied by `make lint`.

## Testing conventions

- Prefer table-driven tests with a shared execution loop; flag duplicated per-case logic.
- Use testify asserts, as the existing suite does; build fixtures through shared helpers.
- Assert observable state, not implementation details.
- Tests must be deterministic and `-race`-clean — `make test` runs with `-race`; flag time-, ordering-, or
  randomness-dependent assertions.
- Tests that need a real model file must skip cleanly when `TEST_MODEL_PATH` is unset (`t.Skip`), never fail.
- Fail fast on setup errors instead of letting them corrupt later assertions.
