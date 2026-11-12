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
