package gguf_parser

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/assert"
)

func TestGGUFFile_Metadata(t *testing.T) {
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

	t.Log("\n", spew.Sdump(f.Metadata()), "\n")
}

func BenchmarkGGUFFile_Metadata(b *testing.B) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		b.Skip("TEST_MODEL_PATH is not set")
		return
	}

	f, err := ParseGGUFFile(mp, UseMMap(), SkipLargeMetadata())
	if err != nil {
		b.Fatal(err)
		return
	}

	b.ReportAllocs()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = f.Metadata()
	}
}

func TestGGUFFile_extractFileType(t *testing.T) {
	ctx := context.Background()

	repo := "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"
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
		t.Run(repo+"/"+tc, func(t *testing.T) {
			gf, err := ParseGGUFFileFromHuggingFace(
				ctx,
				repo,
				fmt.Sprintf("Hermes-2-Pro-Mistral-7B.%s.gguf", tc))
			if err != nil {
				t.Fatal(err)
				return
			}
			md := gf.Metadata()
			ft, ftd := gf.extractFileType(md.Architecture)
			assert.Equal(t, md.FileType.String(), ft.String(), tc+" file type should be equal")
			assert.Equal(t, tc, ftd, tc+" file type descriptor should be equal")
		})
	}

	// Ignore unsupported cases for https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/commit/42f8e463b233df7575f1e1e9a83cb5936db56d2a.
	repo = "Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
	cases = []string{
		"IQ2_M",
		"IQ2_S",
		"IQ2_XS",
		"IQ2_XXS",
		"IQ3_M",
		"IQ3_S",
		"IQ3_XS",
		"IQ3_XXS",
		"IQ4_NL",
		"IQ4_XS",
		// "Q2_K_L",
		"Q2_K_S",
		// "Q3_K_L",
		"Q3_K_M",
		"Q3_K_S",
		"Q4_0",
		// "Q4_0_L",
		"Q4_1",
		// "Q4_1_L",
		// "Q4_K_L",
		"Q4_K_M",
		"Q4_K_S",
		"Q5_0",
		// "Q5_0_L",
		// "Q5_K_L",
		"Q5_K_M",
		"Q5_K_S",
		// "Q6_K_L",
		// "Q6_K_M", == "Q6_K"
		"Q8_0",
	}
	for _, tc := range cases {
		t.Run(repo+"/"+tc, func(t *testing.T) {
			gf, err := ParseGGUFFileFromHuggingFace(
				ctx,
				repo,
				fmt.Sprintf("Qwen2.5-VL-3B-Instruct-%s.gguf", strings.ToLower(tc)))
			if err != nil {
				t.Fatal(err)
				return
			}
			md := gf.Metadata()
			ft, ftd := gf.extractFileType(md.Architecture)
			assert.Equal(t, md.FileType.String(), ft.String(), tc+" file type should be equal")
			assert.Equal(t, tc, ftd, tc+" file type descriptor should be equal")
		})
	}

	repo = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
	cases = []string{
		"BF16",
		"Q2_K",
		"Q2_K_L",
		"Q3_K_M",
		"Q4_K_M",
		"Q5_K_M",
		"Q6_K",
		"Q8_0",
	}
	for _, tc := range cases {
		t.Run(repo+"/"+tc, func(t *testing.T) {
			gf, err := ParseGGUFFileFromHuggingFace(
				ctx,
				repo,
				fmt.Sprintf("DeepSeek-R1-Distill-Qwen-1.5B-%s.gguf", tc))
			if err != nil {
				t.Fatal(err)
				return
			}
			md := gf.Metadata()
			ft, ftd := gf.extractFileType(md.Architecture)
			assert.Equal(t, md.FileType.String(), ft.String(), tc+" file type should be equal")
			assert.Equal(t, tc, ftd, tc+" file type descriptor should be equal")
		})
	}

	repo = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
	cases = []string{
		"IQ1_M",
		"IQ1_S",
		"IQ2_M",
		"IQ2_XXS",
		"IQ3_XXS",
		"IQ4_XS",
		// "Q2_K_XL" == "Q2_K_L"
		// "Q3_K_XL" == "Q3_K_M"
		// "Q4_K_XL" == "Q4_K_M"
	}
	for _, tc := range cases {
		t.Run(repo+"/"+tc, func(t *testing.T) {
			gf, err := ParseGGUFFileFromHuggingFace(
				ctx,
				repo,
				fmt.Sprintf("DeepSeek-R1-Distill-Qwen-1.5B-UD-%s.gguf", tc))
			if err != nil {
				t.Fatal(err)
				return
			}
			md := gf.Metadata()
			ft, ftd := gf.extractFileType(md.Architecture)
			assert.Equal(t, md.FileType.String(), ft.String(), tc+" file type should be equal")
			assert.Equal(t, tc, ftd, tc+" file type descriptor should be equal")
		})
	}
}
