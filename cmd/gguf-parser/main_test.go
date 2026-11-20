package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// The flag definitions carry their own validation through the per-flag Action hook. Testing the
// helper alone cannot see a hook that stopped being wired, so run the real app and let the hook
// reject the llama.cpp word values.
func TestGPULayersFlagsRejectTheLlamaCppAutoValue(t *testing.T) {
	cases := []struct {
		name    string
		flag    string
		wantErr string
	}{
		{
			name:    "the main flag rejects the llama.cpp auto value",
			flag:    "--gpu-layers",
			wantErr: `--gpu-layers -1 is llama.cpp's "auto"`,
		},
		{
			name:    "the draft flag rejects the llama.cpp auto value",
			flag:    "--gpu-layers-draft",
			wantErr: `--gpu-layers-draft -1 is llama.cpp's "auto"`,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			app := newApp("gguf-parser")

			err := app.Run([]string{"gguf-parser", c.flag, "-1"})

			assert.ErrorContains(t, err, c.wantErr)
		})
	}
}

func TestReadLimitFlags(t *testing.T) {
	for _, tc := range []struct{ flag, value string }{
		{"--max-remote-shards", "0"},
		{"--max-remote-shards", "100000"},
		{"--metadata-budget-mib", "0"},
		{"--metadata-budget-mib", "17592186044416"},
		{"--remote-read-timeout", "0s"},
	} {
		t.Run(tc.flag+"="+tc.value, func(t *testing.T) {
			err := newApp("gguf-parser").Run([]string{"gguf-parser", tc.flag, tc.value})
			assert.ErrorContains(t, err, tc.flag)
		})
	}
}
