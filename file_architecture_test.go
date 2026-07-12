package gguf_parser

import (
	"context"
	"os"
	"testing"

	"github.com/davecgh/go-spew/spew"
)

func TestGGUFFile_Architecture(t *testing.T) {
	ctx := context.Background()

	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"NousResearch/Hermes-2-Pro-Mistral-7B-GGUF",
		"Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	t.Log("\n", spew.Sdump(f.Architecture()), "\n")
}

func TestGGUFFile_Architecture_ClipProjectorType(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name     string
		repo     string
		file     string
		expected string
	}{
		// Single-modality projectors declare "clip.projector_type".
		{"vision only", "ggml-org/pixtral-12b-GGUF", "mmproj-pixtral-12b-f16.gguf", "pixtral"},
		// Mixed-modality projectors declare "clip.vision.projector_type" instead,
		// see https://github.com/gpustack/gguf-parser-go/issues/25.
		{"vision and audio", "ggml-org/gemma-4-12B-it-GGUF", "mmproj-gemma-4-12B-it-bf16.gguf", "gemma4uv"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f, err := ParseGGUFFileFromHuggingFace(ctx, tc.repo, tc.file, SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}
			if actual := f.Architecture().ClipProjectorType; actual != tc.expected {
				t.Errorf("ClipProjectorType: got %q, want %q", actual, tc.expected)
			}
		})
	}
}

func BenchmarkGGUFFile_Architecture(b *testing.B) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		b.Skip("TEST_MODEL_PATH is not set")
		return
	}

	f, err := ParseGGUFFile(mp, SkipLargeMetadata(), UseMMap())
	if err != nil {
		b.Fatal(err)
		return
	}

	b.ReportAllocs()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = f.Architecture()
	}
}
