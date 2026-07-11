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

	// The projector must not be estimated with the native image size and
	// an un-merged patch count,
	// see https://github.com/gpustack/gguf-parser-go/issues/21.
	dflt := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	if nonUMA := dflt.VRAMs[0].NonUMA; nonUMA > 2*gib {
		t.Errorf("default estimate: NonUMA VRAM %s exceeds 2 GiB", nonUMA)
	}

	// The visual max image size option must take effect for this projector type.
	smaller := f.EstimateLLaMACppRun(WithLLaMACppVisualMaxImageSize(512)).SummarizeItem(false, 0, 0)
	if smaller.VRAMs[0].NonUMA >= dflt.VRAMs[0].NonUMA {
		t.Errorf("visual max image size 512 estimate: NonUMA VRAM %s is not lower than default %s",
			smaller.VRAMs[0].NonUMA, dflt.VRAMs[0].NonUMA)
	}

	// Unknown or new projector types must be bounded as well,
	// instead of falling through every special case.
	for i := range f.Header.MetadataKV {
		if f.Header.MetadataKV[i].Key == "clip.projector_type" {
			f.Header.MetadataKV[i].Value = "future_projector_type"
		}
	}
	unknown := f.EstimateLLaMACppRun().SummarizeItem(false, 0, 0)
	if nonUMA := unknown.VRAMs[0].NonUMA; nonUMA > 2*gib {
		t.Errorf("unknown projector type estimate: NonUMA VRAM %s exceeds 2 GiB", nonUMA)
	}
}
