package gguf_parser

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/assert"
)

func TestGGUFFile_Model(t *testing.T) {
	ctx := context.Background()

	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"NousResearch/Hermes-2-Pro-Mistral-7B-GGUF",
		"Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf",
		UseApproximate())
	if err != nil {
		t.Fatal(err)
		return
	}

	t.Log("\n", spew.Sdump(f.Model()), "\n")
}

func BenchmarkGGUFFile_Model(b *testing.B) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		b.Skip("TEST_MODEL_PATH is not set")
		return
	}

	f, err := ParseGGUFFile(mp, UseMMap(), UseApproximate())
	if err != nil {
		b.Fatal(err)
		return
	}

	b.ReportAllocs()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = f.Model()
	}
}

func TestGGUFFile_guessFileType(t *testing.T) {
	ctx := context.Background()

	cases := []string{
		"Q2_K",
		"Q3_K_L",
		"Q3_K_M",
		"Q3_K_S",
		"Q4_0",
		"Q4_K_M",
		"Q4_K_S",
		"Q5_0",
		"Q5_K_M",
		"Q5_K_S",
		"Q6_K",
		"Q8_0",
	}
	for _, tc := range cases {
		t.Run(tc, func(t *testing.T) {
			gf, err := ParseGGUFFileFromHuggingFace(
				ctx,
				"NousResearch/Hermes-2-Pro-Mistral-7B-GGUF",
				fmt.Sprintf("Hermes-2-Pro-Mistral-7B.%s.gguf", tc))
			if err != nil {
				t.Fatal(err)
				return
			}
			assert.Equal(t, gf.Model().FileType.String(), gf.guessFileType().String(), tc+" file type should be equal")
		})
	}
}
