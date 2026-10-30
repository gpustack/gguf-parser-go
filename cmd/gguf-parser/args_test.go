package main

import (
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
			name:     "the value auto enables it",
			args:     []string{"gguf-parser", "--flash-attn", "auto"},
			expected: []string{"gguf-parser", "--flash-attn=true"},
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
