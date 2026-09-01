# GGUF Parser

A Go library and CLI (`cmd/gguf-parser`) that parses GGUF model files — local, remote via ranged
HTTP reads, or pulled from the HuggingFace, ModelScope, or Ollama registries — and estimates their
memory usage and maximum tokens per second without downloading entire files.

## Project Structure

- Root package `github.com/gpustack/gguf-parser-go` (package `gguf_parser`) — the library.
  - `file*.go` — parsing (`file.go`, `file_option.go`, `file_from_remote.go`, `file_from_distro.go`), metadata (`file_metadata.go`), architecture detection (`file_architecture.go`), tokenizer (`file_tokenizer.go`), and the estimators.
  - `ggml.go` / `scalar.go` — GGUF types and scalar values.
  - `file_estimate__llamacpp.go` / `file_estimate__stablediffusioncpp.go` — per-backend estimators.
  - `file_estimate_option.go` — estimator options (offload layers, context size, flash attention, mmap, adapters, RPC).
  - `cache.go` — ranged-read caching.
  - `ollama_*.go` — Ollama registry readers.
  - `gen.go` / `gen.*.go` + `zz_generated.*.go` — code generation entry points and generated output.
- `cmd/gguf-parser/` — the CLI; its own `go.mod`, built on urfave/cli/v2, dot-importing the root package.
- `util/` — helper packages: `anyx`, `bytex`, `funcx`, `httpx`, `json`, `osx`, `ptr`, `signalx`, `slicex`, `stringx`.
- `Makefile` — the whole workflow is Makefile-driven; there is no `hack/` directory.
- `.github/workflows/` — CI.
- `.sbin/` — git-ignored downloaded tools (goimports-reviser, golangci-lint, lipo).

## Architecture

The parse pipeline reads the GGUF header, metadata, and tensor list, then detects the architecture
(`file_architecture.go`), builds the tokenizer (`file_tokenizer.go`), and feeds per-backend
estimators (`file_estimate__llamacpp.go` for llama.cpp, `file_estimate__stablediffusioncpp.go` for
stable-diffusion.cpp), which project RAM/VRAM usage and maximum tokens per second from that
metadata. Remote files are read through ranged HTTP requests backed by `cache.go`, so only the
bytes needed for parsing are fetched. Registry readers (`file_from_distro.go`, `ollama_*.go`)
resolve HuggingFace, ModelScope, and Ollama model references into parseable sources.

## Development

- `make deps` — tidy and download modules for both Go modules (root library and `cmd/gguf-parser`); `DEPS_UPDATE=true` upgrades them.
- `make generate` — run the `//go:generate` stages (stringer + regression) in both modules, producing `zz_generated.*.go`.
- `make lint` — goimports-reviser (import order std/general/company/project) plus `golangci-lint run --fix`, in both modules; tools are downloaded into `.sbin/`.
- `make test` — `go test -v -failfast -race -cover` over the root module.
- `make benchmark` — run `Benchmark*` with `-benchmem`.
- `make build` (`make gguf-parser`) — cross-compile the CLI into `.dist/` for darwin/linux/windows.
- `make package` — docker buildx image build (set `PACKAGE_PUBLISH=true` to push).
- `make ci` — deps, generate, lint, test, build (the default target).
- Model-dependent tests read the model from `TEST_MODEL_PATH` and skip when it is unset: `TEST_MODEL_PATH=/path/to/model.gguf make test`.

## Go conventions

- Prefer clarity over cleverness to simplify long-term code maintenance.
- Handle errors explicitly; never use panics for control flow.
- Document exported APIs with behavior, expectations, and constraints; doc comments end in periods (godot).
- Use `any`, not `interface{}` (enforced by gofmt rewrite rules).
- Name multi-word Go source files in snake_case (`file_from_remote.go`, `file_estimate__llamacpp.go`), never flat-concatenated.
- Keep concurrency simple, safe, justified, and minimally applied.
- Respect the linter config: gofumpt with extra rules, 150-character line limit, importas aliasing, import order std/general/company/project.
- Never hand-edit generated files (`zz_generated.*.go`, `gen.*.go`); run `make generate` instead.

## Testing conventions

- Use testify for assertions.
- Keep tests deterministic and `-race` clean; the suite runs with `-race -failfast`.
- Model-dependent tests skip via `t.Skip` when `TEST_MODEL_PATH` is unset.
- Prefer table-driven cases where the shape fits.
