.SILENT:
.DEFAULT_GOAL := ci

SHELL := /bin/bash

SRCDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
GOOS := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
LINT_DIRTY ?= false
VERSION ?= $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '[:upper:]' '[:lower:]' || echo "unknown")

deps:
	@echo "+++ deps +++"

	go mod tidy
	go mod download

	cd "$(SRCDIR)/cmd/gguf-parser" && go mod tidy && go mod download

	@echo "--- deps ---"

generate:
	@echo "+++ generate +++"

	go generate $(SRCDIR)/...

	@echo "--- generate ---"

lint:
	@echo "+++ lint +++"

	if [[ "$(LINT_DIRTY)" == "true" ]]; then \
  		if [[ -n $$(git status --porcelain) ]]; then \
  			echo "Code tree is dirty."; \
  			exit 1; \
  		fi; \
	fi

	[[ -d "$(SRCDIR)/.sbin" ]] || mkdir -p "$(SRCDIR)/.sbin"

	[[ -f "$(SRCDIR)/.sbin/goimports-reviser" ]] || \
		curl --retry 3 --retry-all-errors --retry-delay 3 -sSfL "https://github.com/incu6us/goimports-reviser/releases/download/v3.6.5/goimports-reviser_3.6.5_$(GOOS)_$(GOARCH).tar.gz" \
		| tar -zxvf - --directory "$(SRCDIR)/.sbin" --no-same-owner --exclude ./LICENSE --exclude ./README.md && chmod +x "$(SRCDIR)/.sbin/goimports-reviser"
	go list -f "{{.Dir}}" $(SRCDIR)/... | xargs -I {} find {} -maxdepth 1 -type f -name '*.go' ! -name 'gen.*' ! -name 'zz_generated.*' \
		| xargs -I {} "$(SRCDIR)/.sbin/goimports-reviser" -use-cache -imports-order=std,general,company,project,blanked,dotted -output=file {}

	[[ -f "$(SRCDIR)/.sbin/golangci-lint" ]] || \
		curl --retry 3 --retry-all-errors --retry-delay 3 -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh \
		| sh -s -- -b "$(SRCDIR)/.sbin" "v1.59.0"
	"$(SRCDIR)/.sbin/golangci-lint" run --fix $(SRCDIR)/...

	@echo "--- lint ---"

test:
	@echo "+++ test +++"

	go test -v -failfast -race -cover -timeout=30m $(SRCDIR)/...

	@echo "--- test ---"

benchmark:
	@echo "+++ benchmark +++"

	go test -v -failfast -run="^Benchmark[A-Z]+" -bench=. -benchmem -timeout=30m $(SRCDIR)/...

	@echo "--- benchmark ---"

gguf-parser:
	@echo "+++ gguf-parser +++"
	[[ -d "$(SRCDIR)/.dist" ]] || mkdir -p "$(SRCDIR)/.dist"

	cd "$(SRCDIR)/cmd/gguf-parser" && for os in darwin linux windows; do \
  		tags="netgo"; \
  		if [[ $$os == "windows" ]]; then \
		  suffix=".exe"; \
		  tags="netcgo"; \
		else \
		  suffix=""; \
		fi; \
		for arch in amd64 arm64; do \
		  	echo "Building gguf-parser for $$os-$$arch $(VERSION)"; \
			GOOS="$$os" GOARCH="$$arch" CGO_ENABLED=1 go build \
				-trimpath \
				-ldflags="-w -s -X main.Version=$(VERSION)" \
				-tags="urfave_cli_no_docs $$tags" \
				-o $(SRCDIR)/.dist/gguf-parser-$$os-$$arch$$suffix; \
		done; \
		if [[ $$os == "darwin" ]]; then \
		  [[ -d "$(SRCDIR)/.sbin" ]] || mkdir -p "$(SRCDIR)/.sbin"; \
		  [[ -f "$(SRCDIR)/.sbin/lipo" ]] || \
			GOBIN="$(SRCDIR)/.sbin" go install github.com/konoui/lipo@v0.9.1; \
		  	"$(SRCDIR)/.sbin/lipo" -create -output $(SRCDIR)/.dist/gguf-parser-darwin-universal $(SRCDIR)/.dist/gguf-parser-darwin-amd64 $(SRCDIR)/.dist/gguf-parser-darwin-arm64; \
		fi;\
	done

ci: deps generate test lint

build: gguf-parser
