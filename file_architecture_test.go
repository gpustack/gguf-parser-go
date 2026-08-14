package gguf_parser

import (
	"context"
	"os"
	"slices"
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
		{"vision and audio", "ggml-org/gemma-4-12B-it-GGUF", "mmproj-gemma-4-12B-it-BF16.gguf", "gemma4uv"},
		// Audio-only projectors of the same generation declare "clip.audio.projector_type" only.
		{"audio only", "elizaos/eliza-1", "voice/asr/eliza-1-asr-mmproj.gguf", "qwen3a"},
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

func TestGGUFFile_Architecture_HybridAttention(t *testing.T) {
	ctx := context.Background()

	// Qwen3.6 is hybrid: it interleaves recurrent layers with full (self-)attention layers,
	// and declares the interleaving with "qwen35moe.full_attention_interval",
	// while being none of the architectures hardcoded as hybrid.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"unsloth/Qwen3.6-35B-A3B-GGUF",
		"Qwen3.6-35B-A3B-UD-IQ1_M.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	a := f.Architecture()
	if a.Architecture != "qwen35moe" {
		t.Errorf("Architecture: got %q, want %q", a.Architecture, "qwen35moe")
	}
	if a.BlockCount != 40 {
		t.Errorf("BlockCount: got %d, want 40", a.BlockCount)
	}
	if a.FullAttentionInterval != 4 {
		t.Errorf("FullAttentionInterval: got %d, want 4", a.FullAttentionInterval)
	}
	if !a.AttentionHybrid {
		t.Error("AttentionHybrid: got false, want true")
	}
	if !a.AttentionRecurrent {
		t.Error("AttentionRecurrent: got false, want true")
	}
}

func TestGGUFArchitecture_memoryKindOfLayer(t *testing.T) {
	cases := []struct {
		name string
		arch GGUFArchitecture
		// fullAttention and recurrent hold the expected classification of the layers 0 to 7.
		fullAttention, recurrent []bool
	}{
		{
			name:          "attentive",
			arch:          GGUFArchitecture{BlockCount: 8},
			fullAttention: []bool{true, true, true, true, true, true, true, true},
			recurrent:     []bool{false, false, false, false, false, false, false, false},
		},
		{
			// Mamba and RWKV hold a recurrent state on every layer and no KV cache at all.
			name:          "recurrent",
			arch:          GGUFArchitecture{BlockCount: 8, AttentionRecurrent: true},
			fullAttention: []bool{false, false, false, false, false, false, false, false},
			recurrent:     []bool{true, true, true, true, true, true, true, true},
		},
		{
			// Falcon-H1 has no interleaving to declare, and the one Jamba declares is not read here,
			// so both are charged on every layer,
			// which is what this library did for every hybrid architecture before.
			name:          "hybrid without a declared interleaving",
			arch:          GGUFArchitecture{BlockCount: 8, AttentionRecurrent: true, AttentionHybrid: true},
			fullAttention: []bool{true, true, true, true, true, true, true, true},
			recurrent:     []bool{true, true, true, true, true, true, true, true},
		},
		{
			// Qwen3.5 declares an interval of 4, making every fourth layer a full (self-)attention one.
			name:          "hybrid with a declared interleaving",
			arch:          GGUFArchitecture{BlockCount: 8, AttentionRecurrent: true, AttentionHybrid: true, FullAttentionInterval: 4},
			fullAttention: []bool{false, false, false, true, false, false, false, true},
			recurrent:     []bool{true, true, true, false, true, true, true, false},
		},
		{
			// An interval of one leaves no recurrent layer, so the architecture is not hybrid at all.
			name:          "full attention interval of one",
			arch:          GGUFArchitecture{BlockCount: 8, FullAttentionInterval: 1},
			fullAttention: []bool{true, true, true, true, true, true, true, true},
			recurrent:     []bool{false, false, false, false, false, false, false, false},
		},
		{
			// LFM2 declares its interleaving with zero entries in the per-layer head_count_kv:
			// the zero layers are recurrent (short convolution), the others full (self-)attention.
			name: "hybrid with per-layer KV heads",
			arch: GGUFArchitecture{
				BlockCount: 8, AttentionRecurrent: true, AttentionHybrid: true,
				AttentionHeadCountKVs: []uint64{0, 0, 8, 0, 8, 0, 8, 8},
			},
			fullAttention: []bool{false, false, true, false, true, false, true, true},
			recurrent:     []bool{true, true, false, true, false, true, false, false},
		},
		{
			// The per-layer head_count_kv says which layer is which exactly,
			// so it takes precedence over a declared interval.
			name: "per-layer KV heads over a full attention interval",
			arch: GGUFArchitecture{
				BlockCount: 8, AttentionRecurrent: true, AttentionHybrid: true,
				FullAttentionInterval: 4, AttentionHeadCountKVs: []uint64{8, 0, 0, 0, 8, 0, 0, 0},
			},
			fullAttention: []bool{true, false, false, false, true, false, false, false},
			recurrent:     []bool{false, true, true, true, false, true, true, true},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for i := uint64(0); i < tc.arch.BlockCount; i++ {
				fullAttention, recurrent := tc.arch.memoryKindOfLayer(i)
				if fullAttention != tc.fullAttention[i] {
					t.Errorf("layer %d: fullAttention: got %v, want %v", i, fullAttention, tc.fullAttention[i])
				}
				if recurrent != tc.recurrent[i] {
					t.Errorf("layer %d: recurrent: got %v, want %v", i, recurrent, tc.recurrent[i])
				}
			}
		})
	}
}

func TestGGUFFile_Architecture_PerLayerKVHeads(t *testing.T) {
	ctx := context.Background()

	// LFM2 declares "lfm2.attention.head_count_kv" as a per-layer array:
	// 6 of its 16 layers hold a KV cache with 8 KV heads,
	// the 10 zero layers hold a short convolution state instead.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"LiquidAI/LFM2-1.2B-GGUF",
		"LFM2-1.2B-Q8_0.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	a := f.Architecture()
	if a.Architecture != "lfm2" || a.BlockCount != 16 {
		t.Fatalf("architecture: got %q with %d blocks, want %q with 16 blocks",
			a.Architecture, a.BlockCount, "lfm2")
	}
	wantKVs := []uint64{0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0}
	if !slices.Equal(a.AttentionHeadCountKVs, wantKVs) {
		t.Errorf("AttentionHeadCountKVs: got %v, want %v", a.AttentionHeadCountKVs, wantKVs)
	}
	if a.AttentionHeadCountKV != 8 {
		t.Errorf("AttentionHeadCountKV: got %d, want 8", a.AttentionHeadCountKV)
	}
	if a.ShortConvLCache != 3 {
		t.Errorf("ShortConvLCache: got %d, want 3", a.ShortConvLCache)
	}
	if !a.AttentionHybrid || !a.AttentionRecurrent {
		t.Errorf("AttentionHybrid/AttentionRecurrent: got %v/%v, want true/true",
			a.AttentionHybrid, a.AttentionRecurrent)
	}
	for i, wantAttention := range []bool{false, false, true, false} {
		fullAttention, recurrent := a.memoryKindOfLayer(uint64(i))
		if fullAttention != wantAttention || recurrent == wantAttention {
			t.Errorf("layer %d: got fullAttention=%v recurrent=%v, want fullAttention=%v recurrent=%v",
				i, fullAttention, recurrent, wantAttention, !wantAttention)
		}
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
