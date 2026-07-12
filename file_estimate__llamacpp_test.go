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
		{"without attention head count", "ggml-org/gemma-4-12B-it-GGUF", "mmproj-gemma-4-12B-it-bf16.gguf"},
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
