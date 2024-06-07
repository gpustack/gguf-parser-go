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
