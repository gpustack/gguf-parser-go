package gguf_parser

import (
	"context"
	"os"
	"testing"
	"time"

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
		{
			"bartowski/gemma-2-9b-it-GGUF",
			"gemma-2-9b-it-Q3_K_M.gguf",
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

func TestParseGGUFFileFromModelScope(t *testing.T) {
	ctx := context.Background()

	cases := [][2]string{
		{
			"qwen/Qwen1.5-0.5B-Chat-GGUF",
			"qwen1_5-0_5b-chat-q5_k_m.gguf",
		},
		{
			"HIT-SCIR/huozi3-gguf",
			"huozi3-q2_k.gguf",
		},
		{
			"shaowenchen/chinese-alpaca-2-13b-16k-gguf",
			"chinese-alpaca-2-13b-16k.Q5_K.gguf",
		},
	}
	for _, tc := range cases {
		t.Run(tc[0]+"/"+tc[1], func(t *testing.T) {
			f, err := ParseGGUFFileFromModelScope(ctx, tc[0], tc[1], SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}
			t.Log("\n", spew.Sdump(f), "\n")
		})
	}
}

func TestParseGGUFFileFromOllama(t *testing.T) {
	ctx := context.Background()

	cases := []string{
		"gemma2",
		"llama3.1",
		"qwen2:72b-instruct-q3_K_M",
	}
	for _, tc := range cases {
		t.Run(tc, func(t *testing.T) {
			start := time.Now()
			f, err := ParseGGUFFileFromOllama(ctx, tc, SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}
			t.Logf("cost: %v\n", time.Since(start))
			t.Log("\n", spew.Sdump(f), "\n")
		})
	}
}
