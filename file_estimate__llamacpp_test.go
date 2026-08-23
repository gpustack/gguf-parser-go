package gguf_parser

import (
	"context"
	"testing"

	"github.com/davecgh/go-spew/spew"
)

func TestGGUFFile_EstimateLLaMACppRun(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name  string
		given *GGUFFile
	}{
		{
			name: "mixtral 7B",
			given: func() *GGUFFile {
				f, err := ParseGGUFFileFromHuggingFace(
					ctx,
					"NousResearch/Hermes-2-Pro-Mistral-7B-GGUF",
					"Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf",
					SkipLargeMetadata())
				if err != nil {
					t.Fatal(err)
				}
				return f
			}(),
		},
		{
			name: "mixtral 8x7B",
			given: func() *GGUFFile {
				f, err := ParseGGUFFileFromHuggingFace(
					ctx,
					"NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF",
					"Nous-Hermes-2-Mixtral-8x7B-DPO.Q5_K_M.gguf",
					SkipLargeMetadata())
				if err != nil {
					t.Fatal(err)
				}
				return f
			}(),
		},
		{
			name: "wizardlm 8x22B",
			given: func() *GGUFFile {
				f, err := ParseGGUFFileFromHuggingFace(
					ctx,
					"MaziyarPanahi/WizardLM-2-8x22B-GGUF",
					"WizardLM-2-8x22B.IQ1_M.gguf",
					SkipLargeMetadata())
				if err != nil {
					t.Fatal(err)
				}
				return f
			}(),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.given
			t.Log("\n", spew.Sdump(f.EstimateLLaMACppRun()), "\n")
		})
	}
}

func TestGGUFFile_EstimateLLaMACppRun_ContextSize(t *testing.T) {
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

	cases := []struct {
		name string
		opts []GGUFRunEstimateOption
	}{
		{"1024(fp16)", []GGUFRunEstimateOption{WithLLaMACppContextSize(1024)}},
		{"1024(fp32)", []GGUFRunEstimateOption{WithLLaMACppContextSize(1024), WithLLaMACppCacheKeyType(GGMLTypeF32), WithLLaMACppCacheValueType(GGMLTypeF32)}},
		{"4096(fp16)", []GGUFRunEstimateOption{WithLLaMACppContextSize(4096)}},
		{"4096(fp32)", []GGUFRunEstimateOption{WithLLaMACppContextSize(4096), WithLLaMACppCacheKeyType(GGMLTypeF32), WithLLaMACppCacheValueType(GGMLTypeF32)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("\n", spew.Sdump(f.EstimateLLaMACppRun(tc.opts...)), "\n")
		})
	}
}

func TestGGUFFile_EstimateLLaMACppRun_OffloadLayers(t *testing.T) {
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

	cases := []struct {
		name string
		opts []GGUFRunEstimateOption
	}{
		{"offload 0 layer", []GGUFRunEstimateOption{WithLLaMACppOffloadLayers(0)}},
		{"offload 1 layer", []GGUFRunEstimateOption{WithLLaMACppOffloadLayers(1)}},
		{"offload 10 layers", []GGUFRunEstimateOption{WithLLaMACppOffloadLayers(10)}},
		{"offload all layers", []GGUFRunEstimateOption{}},
		{"offload 33 layers", []GGUFRunEstimateOption{WithLLaMACppOffloadLayers(33)}}, // exceeds the number of layers
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("\n", spew.Sdump(f.EstimateLLaMACppRun(tc.opts...)), "\n")
		})
	}
}

func TestGGUFFile_EstimateLLaMACppRun_Projector(t *testing.T) {
	ctx := context.Background()

	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"noctrex/LightOnOCR-2-1B-GGUF",
		"mmproj-BF16.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	const gib = 1 << 30

	// The projector must not be estimated with the native image size,
	// which charges an 8.7 GiB attention buffer for this one,
	// see https://github.com/gpustack/gguf-parser-go/issues/21.
	dflt := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	if nonUMA := dflt.VRAMs[0].NonUMA; nonUMA > 4*gib {
		t.Errorf("default estimate: NonUMA VRAM %s exceeds 4 GiB", nonUMA)
	}

	// The visual max image size option must take effect for this projector type.
	smaller := f.EstimateLLaMACppRun(WithLLaMACppVisualMaxImageSize(512)).SummarizeItem(false, 0, 0)
	if smaller.VRAMs[0].NonUMA >= dflt.VRAMs[0].NonUMA {
		t.Errorf("visual max image size 512 estimate: NonUMA VRAM %s is not lower than default %s",
			smaller.VRAMs[0].NonUMA, dflt.VRAMs[0].NonUMA)
	}

	// Flash attention must take effect for the projector as well:
	// the clip encoder does not materialize the attention score matrix with it,
	// see https://github.com/gpustack/gguf-parser-go/issues/23.
	fa := f.EstimateLLaMACppRun(WithFlashAttention()).SummarizeItem(false, 0, 0)
	if fa.VRAMs[0].NonUMA >= dflt.VRAMs[0].NonUMA {
		t.Errorf("flash attention estimate: NonUMA VRAM %s is not lower than default %s",
			fa.VRAMs[0].NonUMA, dflt.VRAMs[0].NonUMA)
	}

	// Unknown or new projector types must be bounded as well,
	// instead of falling through every special case.
	for i := range f.Header.MetadataKV {
		if f.Header.MetadataKV[i].Key == "clip.projector_type" {
			f.Header.MetadataKV[i].Value = "future_projector_type"
		}
	}
	unknown := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	if nonUMA := unknown.VRAMs[0].NonUMA; nonUMA > 4*gib {
		t.Errorf("unknown projector type estimate: NonUMA VRAM %s exceeds 4 GiB", nonUMA)
	}
}

func TestGGUFFile_EstimateLLaMACppRun_ProjectorWithoutImageSize(t *testing.T) {
	ctx := context.Background()

	// dots.ocr's projector declares no clip.vision.image_size; assuming zero-pixel images
	// charged nothing for the 42-block encoder, and real usage measured above the estimate.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"ggml-org/dots.ocr-GGUF",
		"mmproj-dots.ocr-f16.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	const gib = 1 << 30
	weights := uint64(f.ModelSize)
	dflt := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	if nonUMA := uint64(dflt.VRAMs[0].NonUMA); nonUMA < weights+gib/2 {
		t.Errorf("default estimate: NonUMA VRAM %s charges almost nothing beyond the %s weights",
			dflt.VRAMs[0].NonUMA, GGUFBytesScalar(weights))
	}
	if nonUMA := dflt.VRAMs[0].NonUMA; nonUMA > 8*gib {
		t.Errorf("default estimate: NonUMA VRAM %s exceeds 8 GiB", nonUMA)
	}
}

func TestGGUFFile_EstimateLLaMACppRun_ProjectorMergedClassEmbedding(t *testing.T) {
	ctx := context.Background()

	// InternVL's encoder carries a class embedding token.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"ggml-org/InternVL2_5-1B-GGUF",
		"mmproj-InternVL2_5-1B-f16.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	// A spatial merge reduces the projector's output tokens, so declaring a larger merge must
	// never increase the estimate. The class embedding is a single position regardless of the
	// merge; multiplying it by the merge factor too would grow the attention buffer
	// quadratically with the merge instead.
	for i := range f.Header.MetadataKV {
		if f.Header.MetadataKV[i].Key == "clip.projector_type" {
			f.Header.MetadataKV[i].Value = "future_projector_type"
		}
	}
	f.Header.MetadataKV = append(f.Header.MetadataKV, GGUFMetadataKV{
		Key:       "clip.vision.spatial_merge_size",
		ValueType: GGUFMetadataValueTypeUint32,
		Value:     uint32(2),
	})
	merged2 := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	// 448px at patch size 14 is 32 patches per side, so a merge of 32 collapses the projector's
	// output to a single token while the encoder still attends over every patch.
	f.Header.MetadataKV[len(f.Header.MetadataKV)-1].Value = uint32(32)
	merged32 := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	if merged32.VRAMs[0].NonUMA > merged2.VRAMs[0].NonUMA {
		t.Errorf("spatial merge 32 estimate: NonUMA VRAM %s exceeds the spatial merge 2 estimate %s",
			merged32.VRAMs[0].NonUMA, merged2.VRAMs[0].NonUMA)
	}
}

func TestGGUFFile_EstimateLLaMACppRun_ProjectorAudioChunked(t *testing.T) {
	ctx := context.Background()

	// LFM2.5-Audio's position table holds 16392 positions but its encoder runs on 1-second
	// chunks (~100 positions); sizing attention over the whole table charged 8.5 GiB for
	// this 0.4 GB projector,
	// see https://github.com/gpustack/gguf-parser-go/issues/26.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"LiquidAI/LFM2.5-Audio-1.5B-GGUF",
		"mmproj-LFM2.5-Audio-1.5B-F16.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	const gib = 1 << 30
	dflt := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	if nonUMA := dflt.VRAMs[0].NonUMA; nonUMA > 2*gib {
		t.Errorf("default estimate: NonUMA VRAM %s exceeds 2 GiB", nonUMA)
	}
}

func TestGGUFFile_EstimateLLaMACppRun_ProjectorConvolutionalEncoder(t *testing.T) {
	ctx := context.Background()

	// Gemma 3n's MobileNetV5 encoder outputs a fixed token grid and its patch size is a
	// convolution stride; treating it as transformer patches charged a ~130 GiB attention
	// buffer for this ~1.4 GB projector.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"Anthonyg5005/gemma-3n-e4b-mmproj-gguf",
		"gemma-3n-mmproj.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	const gib = 1 << 30
	dflt := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	if nonUMA := dflt.VRAMs[0].NonUMA; nonUMA > 3*gib {
		t.Errorf("default estimate: NonUMA VRAM %s exceeds 3 GiB", nonUMA)
	}
}

func TestGGUFFile_EstimateLLaMACppRun_ProjectorFlashAttention(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name string
		repo string
		file string
	}{
		{"vision", "ggml-org/pixtral-12b-GGUF", "mmproj-pixtral-12b-f16.gguf"},
		{"audio", "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF", "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"},
		// Declares no attention head count, so the encoder's attention is not modeled at all;
		// enabling flash attention must not conjure a buffer the estimate did not charge before.
		{"without attention head count", "ggml-org/gemma-4-12B-it-GGUF", "mmproj-gemma-4-12B-it-BF16.gguf"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f, err := ParseGGUFFileFromHuggingFace(ctx, tc.repo, tc.file, SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}

			// Flash attention never costs more than not using it,
			// see https://github.com/gpustack/gguf-parser-go/issues/23.
			dflt := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
			fa := f.EstimateLLaMACppRun(WithFlashAttention()).SummarizeItem(false, 0, 0)
			if fa.VRAMs[0].NonUMA > dflt.VRAMs[0].NonUMA {
				t.Errorf("flash attention estimate: NonUMA VRAM %s exceeds the estimate without it %s",
					fa.VRAMs[0].NonUMA, dflt.VRAMs[0].NonUMA)
			}
		})
	}
}

func TestGGUFFile_EstimateLLaMACppRun_HybridInterleavedKVCache(t *testing.T) {
	ctx := context.Background()

	// Qwen3.6 interleaves 30 recurrent layers with 10 full (self-)attention layers over its 40 blocks,
	// declaring the interleaving with "qwen35moe.full_attention_interval" = 4.
	// llama.cpp caches a KV for the full (self-)attention layers only,
	// and a recurrent state for the recurrent layers only,
	// see https://github.com/ggml-org/llama.cpp/blob/272700b360944e40816a7ea13da8cd723119000a/src/llama-model.cpp#L2176-L2182.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"unsloth/Qwen3.6-35B-A3B-GGUF",
		"Qwen3.6-35B-A3B-UD-IQ1_M.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}
	if a := f.Architecture(); a.Architecture != "qwen35moe" || a.BlockCount != 40 || a.FullAttentionInterval != 4 {
		t.Fatalf("architecture: got %q with %d blocks at an interval of %d, want %q with 40 blocks at an interval of 4",
			a.Architecture, a.BlockCount, a.FullAttentionInterval, "qwen35moe")
	}

	// declareInterval rewrites the declared interval, or drops the key declaring it.
	declareInterval := func(interval uint32, declared bool) {
		kvs := make(GGUFMetadataKVs, 0, len(f.Header.MetadataKV))
		for _, kv := range f.Header.MetadataKV {
			if kv.Key == "qwen35moe.full_attention_interval" {
				if !declared {
					continue
				}
				kv.Value = interval
			}
			kvs = append(kvs, kv)
		}
		f.Header.MetadataKV = kvs
	}
	// Full offload places every block on the single GPU device.
	kvCache := func() uint64 {
		return uint64(f.EstimateLLaMACppRun(WithLLaMACppContextSize(131072)).Devices[1].KVCache.Sum())
	}

	// Each full (self-)attention layer caches 2 kv heads * (256 key + 256 value) head size *
	// 131072 context * 2 bytes, which is 256 MiB, so the 10 of them cache 2.5 GiB,
	// plus about 2 MiB of context independent state for each of the 30 recurrent layers.
	const mib = 1 << 20
	got := kvCache()
	if got < 2500*mib || got > 2700*mib {
		t.Errorf("KV cache at an interval of 4: got %s, want within [2500 MiB, 2700 MiB]", GGUFBytesScalar(got))
	}

	// Splitting the blocks over three GPU devices splits the interleaved KV cache with them,
	// while not offloading it puts all of it back on the host device.
	split := f.EstimateLLaMACppRun(WithLLaMACppContextSize(131072), WithTensorSplitFraction([]float64{1.0 / 3, 2.0 / 3, 1}))
	var splitted uint64
	for i := range split.Devices[1:] {
		if split.Devices[i+1].KVCache.Sum() == 0 {
			t.Errorf("KV cache of the device %d: got 0, want more", i+1)
		}
		splitted += uint64(split.Devices[i+1].KVCache.Sum())
	}
	if splitted != got {
		t.Errorf("KV cache over three devices: got %s, want %s", GGUFBytesScalar(splitted), GGUFBytesScalar(got))
	}
	hosted := f.EstimateLLaMACppRun(WithLLaMACppContextSize(131072), WithoutLLaMACppOffloadKVCache())
	if uint64(hosted.Devices[0].KVCache.Sum()) != got {
		t.Errorf("KV cache without offloading: got %s, want %s", hosted.Devices[0].KVCache.Sum(), GGUFBytesScalar(got))
	}

	// Halving the declared interval doubles the full (self-)attention layers, and so the KV cache.
	declareInterval(2, true)
	if halved := kvCache(); halved < 5000*mib || halved > 5400*mib {
		t.Errorf("KV cache at an interval of 2: got %s, want within [5000 MiB, 5400 MiB]", GGUFBytesScalar(halved))
	}

	// Dropping the key leaves the estimate as it was before any interleaving was parsed:
	// a KV cache charged to all 40 blocks, 10 GiB of it.
	declareInterval(0, false)
	if undeclared := kvCache(); undeclared < 10*1024*mib || undeclared > 10*1024*mib+8*mib {
		t.Errorf("KV cache without a declared interval: got %s, want about 10 GiB", GGUFBytesScalar(undeclared))
	}
}

func TestGGUFFile_EstimateLLaMACppRun_PerLayerKVHeadsKVCache(t *testing.T) {
	ctx := context.Background()

	// LFM2 marks its 10 short-convolution layers with zero entries in the per-layer
	// "lfm2.attention.head_count_kv"; the other 6 layers hold a KV cache with 8 KV heads.
	// llama.cpp allocates the two caches from complementary per-layer filters,
	// see https://github.com/ggml-org/llama.cpp/blob/16d222fc5f2c0fdc3d0180e0b772516ec6e2eddd/src/models/lfm2.cpp#L9-L11.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"LiquidAI/LFM2-1.2B-GGUF",
		"LFM2-1.2B-Q8_0.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	// Full offload places every block on the single GPU device.
	kvCache := func(contextSize int32) uint64 {
		return uint64(f.EstimateLLaMACppRun(WithLLaMACppContextSize(contextSize)).Devices[1].KVCache.Sum())
	}

	// Each of the 6 attention layers caches 8 kv heads * (64 key + 64 value) head size *
	// 32768 context * 2 bytes, which is 64 MiB; the 10 short-convolution layers hold
	// n_embd * (l_cache - 1) floats each, about 16 KiB, independent of the context.
	const mib = 1 << 20
	got := kvCache(32768)
	if got < 384*mib || got > 386*mib {
		t.Errorf("KV cache at 32768 context: got %s, want within [384 MiB, 386 MiB]", GGUFBytesScalar(got))
	}

	// An eighth of the context caches an eighth of the KV, over the same constant state.
	if gotSmall := kvCache(4096); gotSmall < 48*mib || gotSmall > 50*mib {
		t.Errorf("KV cache at 4096 context: got %s, want within [48 MiB, 50 MiB]", GGUFBytesScalar(gotSmall))
	}

	// Splitting the blocks over two GPU devices splits the interleaved KV cache with them,
	// while not offloading it puts all of it back on the host device.
	split := f.EstimateLLaMACppRun(WithLLaMACppContextSize(32768), WithTensorSplitFraction([]float64{0.5, 1}))
	var splitted uint64
	for i := range split.Devices[1:] {
		if split.Devices[i+1].KVCache.Sum() == 0 {
			t.Errorf("KV cache of the device %d: got 0, want more", i+1)
		}
		splitted += uint64(split.Devices[i+1].KVCache.Sum())
	}
	if splitted != got {
		t.Errorf("KV cache over two devices: got %s, want %s", GGUFBytesScalar(splitted), GGUFBytesScalar(got))
	}
	hosted := f.EstimateLLaMACppRun(WithLLaMACppContextSize(32768), WithoutLLaMACppOffloadKVCache())
	if uint64(hosted.Devices[0].KVCache.Sum()) != got {
		t.Errorf("KV cache without offloading: got %s, want %s", hosted.Devices[0].KVCache.Sum(), GGUFBytesScalar(got))
	}
}

func TestGGUFFile_EstimateLLaMACppRun_SlidingWindowKVCache(t *testing.T) {
	ctx := context.Background()

	const mib = 1 << 20
	cases := []struct {
		name string
		repo string
		file string

		// kvCacheMin/kvCacheMax bound the KV cache of the fully offloaded device
		// at a context of 32768 with the default f16 cache.
		kvCacheMin, kvCacheMax uint64
	}{
		{
			// gpt-oss-20b holds a full KV cache on 12 of its 24 layers (64 MiB each at 32768)
			// and a 128-token window on the other 12 (about 4.3 MiB each).
			name: "gpt-oss", repo: "ggml-org/gpt-oss-20b-GGUF", file: "gpt-oss-20b-MXFP4.gguf",
			kvCacheMin: 815 * mib, kvCacheMax: 825 * mib,
		},
		{
			// gemma-3n-E4B holds a full KV cache on 4 of its first 20 layers (64 MiB each at 32768),
			// a 512-token window on the other 16 (5 MiB each), and nothing on the remaining 15,
			// which reuse the leading caches.
			name: "gemma3n", repo: "ggml-org/gemma-3n-E4B-it-GGUF", file: "gemma-3n-E4B-it-Q8_0.gguf",
			kvCacheMin: 330 * mib, kvCacheMax: 342 * mib,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f, err := ParseGGUFFileFromHuggingFace(ctx, tc.repo, tc.file, SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}
			got := uint64(f.EstimateLLaMACppRun(WithLLaMACppContextSize(32768)).Devices[1].KVCache.Sum())
			if got < tc.kvCacheMin || got > tc.kvCacheMax {
				t.Errorf("KV cache at 32768 context: got %s, want within [%s, %s]",
					GGUFBytesScalar(got), GGUFBytesScalar(tc.kvCacheMin), GGUFBytesScalar(tc.kvCacheMax))
			}
			// The full-size SWA cache option restores a full KV cache on every windowed layer.
			full := uint64(f.EstimateLLaMACppRun(WithLLaMACppContextSize(32768), WithLLaMACppFullSizeSWACache()).Devices[1].KVCache.Sum())
			if full <= got {
				t.Errorf("full-size SWA cache: got %s, want more than %s", GGUFBytesScalar(full), GGUFBytesScalar(got))
			}
		})
	}
}

func TestGGUFFile_EstimateLLaMACppRun_NextNPredictLayersKVCache(t *testing.T) {
	ctx := context.Background()

	// The hybrid family filters its NextN/MTP blocks out of the main context cache,
	// see https://github.com/ggml-org/llama.cpp/blob/16d222fc5f2c0fdc3d0180e0b772516ec6e2eddd/src/llama-model.cpp#L2290-L2296.
	// Qwen3.6 declares no NextN blocks; declaring one must uncharge exactly the last
	// layer, a full (self-)attention one at an interval of 4 over 64 blocks.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"unsloth/Qwen3.6-27B-GGUF",
		"Qwen3.6-27B-Q4_K_M.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	kvCache := func() uint64 {
		return uint64(f.EstimateLLaMACppRun(WithLLaMACppContextSize(32768)).Devices[1].KVCache.Sum())
	}
	before := kvCache()

	f.Header.MetadataKV = append(f.Header.MetadataKV, GGUFMetadataKV{
		Key:       "qwen35.nextn_predict_layers",
		ValueType: GGUFMetadataValueTypeUint32,
		Value:     uint32(1),
	})
	after := kvCache()

	// One less full (self-)attention layer: 4 KV heads * (256 key + 256 value) head size *
	// 32768 context * 2 bytes.
	const wantDelta = 4 * (256 + 256) * 32768 * 2
	if before-after != wantDelta {
		t.Errorf("KV cache delta for one NextN block: got %s, want %s",
			GGUFBytesScalar(before-after), GGUFBytesScalar(uint64(wantDelta)))
	}
}

func TestGGUFFile_EstimateLLaMACppRun_LFM2SlidingWindowKVCache(t *testing.T) {
	ctx := context.Background()

	// LFM2 windows its attention layers when it declares a sliding window; the shipped
	// LFM2-1.2B declares none, so declare one and the 6 attention layers must shrink from
	// full context rows to windowed rows while the conv layers stay recurrent,
	// see https://github.com/ggml-org/llama.cpp/blob/16d222fc5f2c0fdc3d0180e0b772516ec6e2eddd/src/models/lfm2.cpp#L24-L29.
	f, err := ParseGGUFFileFromHuggingFace(
		ctx,
		"LiquidAI/LFM2-1.2B-GGUF",
		"LFM2-1.2B-Q8_0.gguf",
		SkipLargeMetadata())
	if err != nil {
		t.Fatal(err)
		return
	}

	kvCache := func() uint64 {
		return uint64(f.EstimateLLaMACppRun(WithLLaMACppContextSize(32768)).Devices[1].KVCache.Sum())
	}
	before := kvCache()

	f.Header.MetadataKV = append(f.Header.MetadataKV, GGUFMetadataKV{
		Key:       "lfm2.attention.sliding_window",
		ValueType: GGUFMetadataValueTypeUint32,
		Value:     uint32(4096),
	})
	a := f.Architecture()
	swaLayers := 0
	for i := uint64(0); i < a.BlockCount; i++ {
		if a.isSWALayer(i) {
			swaLayers++
		}
	}
	if swaLayers != 6 {
		t.Errorf("sliding window layers: got %d, want the 6 attention layers", swaLayers)
	}

	// 6 layers * 512 K elems * 2 (K+V) * 2 bytes over 6144 window rows
	// (4096 window + 2048 logical batch), plus the constant conv state.
	const mib = 1 << 20
	after := kvCache()
	if after >= before || after < 72*mib || after > 74*mib {
		t.Errorf("windowed KV cache: got %s, want within [72 MiB, 74 MiB] and below %s",
			GGUFBytesScalar(after), GGUFBytesScalar(before))
	}
}

func TestGGUFFile_EstimateLLaMACppRun_OutputBufferStaysOnHost(t *testing.T) {
	// llama_context::output_reserve allocates the logits buffer from
	// ggml_backend_cpu_buffer_type, or from the output device's host buffer type,
	// which is pinned system memory rather than VRAM. Offloading every layer must
	// therefore not move that buffer onto the card. Charging it to the device
	// reported roughly 700 MiB of VRAM that no card holds, which is enough to
	// reject a model that fits.
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

	const ctxSize, batchSize = 4096, 512
	a := f.Architecture()
	// nOutputs is min(context, physical batch); the buffer is float32.
	nOutputs := uint64(batchSize)
	wantAtLeast := (a.EmbeddingLength + a.VocabularyLength) * nOutputs * 4

	base := []GGUFRunEstimateOption{
		WithLLaMACppContextSize(ctxSize),
		WithLLaMACppLogicalBatchSize(batchSize),
		WithLLaMACppPhysicalBatchSize(batchSize),
	}

	cases := []struct {
		name   string
		layers uint64
	}{
		{"at the block count", a.BlockCount},
		{"past the block count", a.BlockCount + 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := f.EstimateLLaMACppRun(append(append([]GGUFRunEstimateOption{}, base...),
				WithLLaMACppOffloadLayers(tc.layers))...)
			if got := uint64(e.Devices[0].Footprint); got < wantAtLeast {
				t.Errorf("host footprint %d is below the output buffer alone (%d): the buffer was charged to a device",
					got, wantAtLeast)
			}
		})
	}
}
