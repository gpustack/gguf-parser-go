package gguf_parser

import (
	"context"
	"os"
	"testing"

	"github.com/davecgh/go-spew/spew"
)

func TestParseGGUFFile(t *testing.T) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		t.Skip("TEST_MODEL_PATH is not set")
		return
	}

	// Slow read.
	{
		f, err := ParseGGUFFile(mp)
		if err != nil {
			t.Fatal(err)
			return
		}
		s := spew.ConfigState{
			Indent:   "  ",
			MaxDepth: 5, // Avoid console overflow.
		}
		t.Log("\n", s.Sdump(f), "\n")
	}

	// Fast read.
	{
		f, err := ParseGGUFFile(mp, SkipLargeMetadata(), UseMMap())
		if err != nil {
			t.Fatal(err)
			return
		}
		t.Log("\n", spew.Sdump(f), "\n")
	}
}

func BenchmarkParseGGUFFileMMap(b *testing.B) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		b.Skip("TEST_MODEL_PATH is not set")
		return
	}

	b.ReportAllocs()

	b.ResetTimer()
	b.Run("Normal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFile(mp)
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})

	b.ResetTimer()
	b.Run("UseMMap", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFile(mp, UseMMap())
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})
}

func BenchmarkParseGGUFFileSkipLargeMetadata(b *testing.B) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		b.Skip("TEST_MODEL_PATH is not set")
		return
	}

	b.ReportAllocs()

	b.ResetTimer()
	b.Run("Normal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFile(mp, UseMMap())
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})

	b.ResetTimer()
	b.Run("SkipLargeMetadata", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFile(mp, SkipLargeMetadata(), UseMMap())
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})
}

func TestParseGGUFFileRemote(t *testing.T) {
	const u = "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF" +
		"/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf"

	ctx := context.Background()

	// Slow read.
	{
		f, err := ParseGGUFFileRemote(ctx, u, UseDebug())
		if err != nil {
			t.Fatal(err)
			return
		}
		s := spew.ConfigState{
			Indent:   "  ",
			MaxDepth: 5, // Avoid console overflow.
		}
		t.Log("\n", s.Sdump(f), "\n")
	}

	// Fast read.
	{
		f, err := ParseGGUFFileRemote(ctx, u, UseDebug(), SkipLargeMetadata())
		if err != nil {
			t.Fatal(err)
			return
		}
		t.Log("\n", spew.Sdump(f), "\n")
	}
}

func BenchmarkParseGGUFFileRemoteWithBufferSize(b *testing.B) {
	const u = "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF" +
		"/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf"

	ctx := context.Background()

	b.ReportAllocs()

	b.ResetTimer()
	b.Run("256KibBuffer", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFileRemote(ctx, u, SkipLargeMetadata(), UseBufferSize(256*1024))
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})

	b.ResetTimer()
	b.Run("1MibBuffer", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFileRemote(ctx, u, SkipLargeMetadata(), UseBufferSize(1024*1024))
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})

	b.ResetTimer()
	b.Run("4MibBuffer", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFileRemote(ctx, u, SkipLargeMetadata(), UseBufferSize(4*1024*1024))
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})
}

func TestParseGGUFFileFromHuggingFace(t *testing.T) {
	ctx := context.Background()

	cases := [][2]string{
		{
			"TheBloke/Llama-2-13B-chat-GGUF",
			"llama-2-13b-chat.Q8_0.gguf",
		},
		{
			"lmstudio-community/Yi-1.5-9B-Chat-GGUF",
			"Yi-1.5-9B-Chat-Q5_K_M.gguf",
		},
	}
	for _, tc := range cases {
		t.Run(tc[0]+"/"+tc[1], func(t *testing.T) {
			f, err := ParseGGUFFileFromHuggingFace(ctx, tc[0], tc[1], SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}
			t.Log("\n", spew.Sdump(f), "\n")
		})
	}
}
