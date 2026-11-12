---
name: gguf-parser-go-release
description: Cut a gguf-parser release end to end (push main, select and push the version tag, curate the GitHub Release notes that cmd.yml creates with the binaries and the Docker image), and keep the README version claims aligned with what shipped.
---

# gguf-parser-go release

Two maintainer duties:

1. **Cut a release** — push `main` and a version tag; `.github/workflows/cmd.yml` builds the
   binaries, creates the GitHub Release carrying them, and publishes the Docker image.
2. **Align version docs** — the root `README.md` "Notes" bullets and the
   `cmd/gguf-parser/README.md` help block must match the shipped version line before the tag lands.

## Cut a release

The git tag is the single source of truth: `cmd.yml` injects the tag into the build (`VERSION`) and
keys the release on it. There is no version file to bump and no `release.yml` — the `CMD` workflow
IS the pipeline. A `v*.*.*` tag push builds `.dist` for darwin/linux/windows, creates the GitHub
Release with the seven binaries (prerelease when the tag contains `rc`), then pushes the multi-arch
Docker image (`gpustack/gguf-parser`, semver tags plus the moving `latest` on finals). A plain
`main` push builds branch-level artifacts only — no Release.

The automated Release is born with an empty body. Curating its notes is a maintainer duty, not an
optional polish.

### Hard rules

- Tag shapes: `vX.Y.Z` finals; `rc` inside the tag marks the prerelease automatically. Tags are
  lightweight and point at the release commit on `main` — the history carries no annotated tags;
  do not introduce them.
- A semver tag whose binaries and image are out is burned. NEVER delete or move it; fix forward
  with a new version.
- Release from `origin/main` HEAD.
- Docs-only pushes (`.md`, `docs/**`) skip `ci.yml` and `cmd.yml` through their paths-ignore
  lists: a release train that rides only docs commits runs no CI on `main`. This is the expected
  fast path for a direct-to-main release, not a failure.
- Direct-to-`main` without a pull request is the maintainer mode for this skill; say so in the
  summary. Do not babysit runs unless asked: report the run URL and verify the release afterwards.
- The `cmd/gguf-parser/README.md` GLOBAL OPTIONS block is a shipped doc: it must match `--help`
  byte for byte before a release.

### 1. Preflight (read-only)

```sh
gh auth status
git fetch origin --tags --prune
git status --porcelain                            # must be empty
git rev-parse --abbrev-ref HEAD                   # must be main
git diff origin/main --stat                       # must be empty
git tag -l 'v*' --sort=-v:refname | head -5
```

### 2. Version and release docs (confirm)

- Read the range: `git log --no-merges --oneline <last>..origin/main`.
- Propose the bump from the Conventional Commits in range: any `feat` → minor; fixes, docs, and
  chores → patch.
- The root `README.md` "Notes" must already carry the `Since <VER>` bullet for the chosen version,
  newest first, and `cmd/gguf-parser/README.md` must carry the matching claims. If they are
  missing, land them as a docs commit in this train — docs-only pushes run no CI.
- Confirm the (version, commit) pair before pushing.

### 3. Draft notes

Group the range by type: Features / Fixes / Docs / Other; omit empty sections. Link PRs from
`(#NNN)` subjects. Do not open the body with a heading that repeats the tag. End with the Full
Changelog compare URL (`https://github.com/gpustack/gguf-parser-go/compare/<last>...<VER>`). Keep
this draft for phase 5.

### 4. Push main and the tag (confirm)

```sh
git push origin main                              # any release-riding commits
git tag "$VER" "$SHA"                             # lightweight, repo convention
git push origin "$VER"
gh run list --workflow cmd.yml --limit 3          # the run for the tag
```

The `CMD` run builds the binaries, creates the Release with them, and publishes the image. Direct
mode reports the run URL and moves on; interactive mode may `gh run watch <run-id> --exit-status`.

### 5. Curate the release (confirm)

```sh
gh release view "$VER" --repo gpustack/gguf-parser-go     # appears when the build job finishes
gh release edit "$VER" --repo gpustack/gguf-parser-go --notes-file notes.md
gh release view "$VER" --repo gpustack/gguf-parser-go --json assets --jq '[.assets[].name]'
```

The asset list must carry the seven names: `gguf-parser-darwin-amd64`,
`gguf-parser-darwin-arm64`, `gguf-parser-darwin-universal`, `gguf-parser-linux-amd64`,
`gguf-parser-linux-arm64`, `gguf-parser-windows-amd64.exe`, `gguf-parser-windows-arm64.exe`. In
direct mode, poll briefly for the Release; if the run is still building, hand the user the
curation command instead of waiting.

## Align version docs

`README.md` "Notes" is newest-first with one `Since <VER>` bullet per shipped change of contract;
`cmd/gguf-parser/README.md` owns the usage block (synced from `--help`), the Parameter
Compatibility tables, and the per-feature sections. When a release ships a behavior change, both
must already claim it under the released version — check this in phase 2, not after the tag.

## Scope

Tags, Releases, release notes, and version docs run here. Image production is `cmd.yml`; tests
are `ci.yml`. There is no `release.yml` — never invent one in this flow; the tag push is the
release.
