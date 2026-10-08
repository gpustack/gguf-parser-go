# Copilot Code Review — GGUF Parser

GGUF Parser is a Go library and CLI that parses
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model files — the binary format used by
GGML-based executors such as llama.cpp — and estimates their memory usage and maximum tokens per second (TPS)
without downloading whole files. Estimates track llama.cpp's allocation behavior; deviation from actual usage is
typically around 100MiB. Read `README.md` first for the estimation semantics a PR touches.

[AGENTS.md](../AGENTS.md) owns the project layout, development targets, hard invariants, and the Go and testing
conventions; those facts live there and are not repeated here. Review against it: flag a hard-invariant
violation as a required change, and flag Go or testing changes that break its conventions.

## Out of scope — do not review

- Files matching `zz_generated*`, `gen.*` — generated code and its generators' build-tagged scaffolding.
- `.sbin/` (downloaded lint tooling), `.dist/` (build artifacts).
- Vendored or downloaded tooling of any kind.

Keep feedback specific and actionable; cite the file and line.
