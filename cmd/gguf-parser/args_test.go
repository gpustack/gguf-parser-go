package main

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeArgs(t *testing.T) {
	cases := []struct {
		name     string
		args     []string
		expected []string
	}{
		{
			name:     "the llama.cpp value form with a separate token",
			args:     []string{"gguf-parser", "--flash-attn", "on", "--parallel", "1"},
			expected: []string{"gguf-parser", "--flash-attn=true", "--parallel", "1"},
		},
		{
			name:     "the llama.cpp value form attached to the flag",
			args:     []string{"gguf-parser", "--flash-attn=on", "--parallel", "1"},
			expected: []string{"gguf-parser", "--flash-attn=true", "--parallel", "1"},
		},
		{
			name:     "the llama.cpp value form on an alias",
			args:     []string{"gguf-parser", "-fa", "on"},
			expected: []string{"gguf-parser", "-fa=true"},
		},
		{
			name:     "the stable diffusion cpp alias",
			args:     []string{"gguf-parser", "--diffusion-fa", "off"},
			expected: []string{"gguf-parser", "--diffusion-fa=false"},
		},
		{
			name:     "the value off disables it",
			args:     []string{"gguf-parser", "--flash-attn", "off"},
			expected: []string{"gguf-parser", "--flash-attn=false"},
		},
		{
			name:     "the value auto disables it",
			args:     []string{"gguf-parser", "--flash-attn", "auto"},
			expected: []string{"gguf-parser", "--flash-attn=false"},
		},
		{
			name:     "the value minus one disables it",
			args:     []string{"gguf-parser", "--flash-attn", "-1"},
			expected: []string{"gguf-parser", "--flash-attn=false"},
		},
		{
			name:     "the value enabled enables it",
			args:     []string{"gguf-parser", "--flash-attn", "enabled"},
			expected: []string{"gguf-parser", "--flash-attn=true"},
		},
		{
			name:     "the value disabled disables it",
			args:     []string{"gguf-parser", "--flash-attn", "disabled"},
			expected: []string{"gguf-parser", "--flash-attn=false"},
		},
		{
			name:     "the value one enables it",
			args:     []string{"gguf-parser", "--flash-attn", "1"},
			expected: []string{"gguf-parser", "--flash-attn=true"},
		},
		{
			name:     "the value zero disables it",
			args:     []string{"gguf-parser", "--flash-attn", "0"},
			expected: []string{"gguf-parser", "--flash-attn=false"},
		},
		{
			name:     "the value is case insensitive",
			args:     []string{"gguf-parser", "--flash-attn", "ON"},
			expected: []string{"gguf-parser", "--flash-attn=true"},
		},
		{
			name:     "the bare flag is left alone",
			args:     []string{"gguf-parser", "--flash-attn", "--parallel", "1"},
			expected: []string{"gguf-parser", "--flash-attn", "--parallel", "1"},
		},
		{
			name:     "the GGUF Parser value form is left alone",
			args:     []string{"gguf-parser", "--flash-attn=true", "--flash-attn=false"},
			expected: []string{"gguf-parser", "--flash-attn=true", "--flash-attn=false"},
		},
		{
			name:     "the disabling flag is left alone",
			args:     []string{"gguf-parser", "--no-flash-attn", "--no-flash-attention"},
			expected: []string{"gguf-parser", "--no-flash-attn", "--no-flash-attention"},
		},
		{
			name:     "an unknown value is left alone",
			args:     []string{"gguf-parser", "--flash-attn=maybe"},
			expected: []string{"gguf-parser", "--flash-attn=maybe"},
		},
		{
			name:     "a token that is not a value is left alone",
			args:     []string{"gguf-parser", "--flash-attn", "model.gguf"},
			expected: []string{"gguf-parser", "--flash-attn", "model.gguf"},
		},
		{
			name:     "the other flags are left alone",
			args:     []string{"gguf-parser", "--ctx-size", "24576", "--parallel", "1", "--gpu-layers", "-1"},
			expected: []string{"gguf-parser", "--ctx-size", "24576", "--parallel", "1", "--gpu-layers", "-1"},
		},
		{
			name:     "an empty argument list stays empty",
			args:     []string{},
			expected: []string{},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			assert.Equal(t, c.expected, normalizeArgs(c.args))
		})
	}
}

func TestNormalizeArgsDoesNotChangeTheInput(t *testing.T) {
	args := []string{"gguf-parser", "--flash-attn", "on", "--parallel", "1"}
	normalized := normalizeArgs(args)

	assert.Equal(t, []string{"gguf-parser", "--flash-attn", "on", "--parallel", "1"}, args)
	assert.Equal(t, []string{"gguf-parser", "--flash-attn=true", "--parallel", "1"}, normalized)
}

func TestValidateGPULayers(t *testing.T) {
	cases := []struct {
		name    string
		flag    string
		layers  int
		wantErr string
	}{
		{
			name:   "the llama.cpp all value passes",
			flag:   "gpu-layers",
			layers: -2,
		},
		{
			name:   "zero passes",
			flag:   "gpu-layers",
			layers: 0,
		},
		{
			name:   "a positive count passes",
			flag:   "gpu-layers",
			layers: 33,
		},
		{
			name:    "the llama.cpp auto value is rejected",
			flag:    "gpu-layers",
			layers:  -1,
			wantErr: `--gpu-layers -1 is llama.cpp's "auto"`,
		},
		{
			name:    "a value below the word values is rejected",
			flag:    "gpu-layers",
			layers:  -3,
			wantErr: "invalid --gpu-layers -3",
		},
		{
			name:    "the draft flag carries its own name in the message",
			flag:    "gpu-layers-draft",
			layers:  -1,
			wantErr: `--gpu-layers-draft -1 is llama.cpp's "auto"`,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := validateGPULayers(c.flag, c.layers)
			if c.wantErr == "" {
				assert.NoError(t, err)
				return
			}
			assert.ErrorContains(t, err, c.wantErr)
		})
	}
}

func TestDraftOffloadOverride(t *testing.T) {
	cases := []struct {
		name   string
		draft  int
		isSet  bool
		wantOK bool
		wantN  uint64
	}{
		{
			name:   "an exact draft count restates itself",
			draft:  5,
			isSet:  true,
			wantOK: true,
			wantN:  5,
		},
		{
			name:   "zero restates itself",
			draft:  0,
			isSet:  true,
			wantOK: true,
			wantN:  0,
		},
		{
			name:   "an explicit all overrides the inherited main value",
			draft:  -2,
			isSet:  true,
			wantOK: true,
			wantN:  math.MaxUint64,
		},
		{
			name:  "an unset draft inherits the main value",
			draft: -2,
			isSet: false,
		},
		{
			name:  "a draft value the flag never carried inherits too",
			draft: -1,
			isSet: false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			n, ok := draftOffloadOverride(c.draft, c.isSet)
			assert.Equal(t, c.wantOK, ok)
			if c.wantOK {
				assert.Equal(t, c.wantN, n)
			}
		})
	}
}
