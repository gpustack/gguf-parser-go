package gguf_parser

import (
	"github.com/thxcode/gguf-parser-go/util/ptr"
)

type (
	// GGUFEstimate represents the estimated result of the GGUF file.
	GGUFEstimate struct {
		// ModelWeight is the memory usage of model weight.
		ModelWeight GGUFBytesScalar `json:"modelWeight"`
		// KVCache is the usage of key-value cache.
		KVCache GGUFKVCacheUsage `json:"kvCache"`
		// ComputationGraphOverhead is the overhead of computation graph.
		ComputationGraphOverhead GGUFBytesScalar `json:"computationGraphOverhead"`
		// Others is the trivial usage.
		Others GGUFBytesScalar `json:"others"`
	}

	// GGUFKVCacheUsage represents the usage of kv-cache.
	GGUFKVCacheUsage struct {
		// Key is the memory usage of the cached key.
		Key GGUFBytesScalar `json:"key"`
		// Value is the memory usage of the cached value.
		Value GGUFBytesScalar `json:"value"`
	}
)

// Estimate returns the inference usage estimated result of the GGUF file.
func (gf *GGUFFile) Estimate(opts ...GGUFEstimateOption) (ge GGUFEstimate) {
	var o _GGUFEstimateOptions
	for _, opt := range opts {
		opt(&o)
	}

	a := gf.Architecture()

	contextSize := a.MaximumContextLength
	if o.ContextSize != nil {
		contextSize = uint64(*o.ContextSize)
	}

	// Model weight.
	ge.ModelWeight = gf.ModelSize

	// KV cache,
	// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L2479-L2501.
	{
		kt, vt := GGMLTypeF16, GGMLTypeF16
		kvSize := contextSize
		if o.CacheKeyType != nil {
			kt = *o.CacheKeyType
		}
		if o.CacheValueType != nil {
			vt = *o.CacheValueType
		}
		if a.Architecture == "mamba" {
			// See https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L16122-L16129.
			kt, vt = GGMLTypeF32, GGMLTypeF32
			kvSize = uint64(ptr.Deref(o.ParallelSize, 1))
		}

		var (
			embedKeyGQA = uint64(a.AttentionKeyLength) * a.AttentionHeadCountKV
			embedValGQA = uint64(a.AttentionValueLength) * a.AttentionHeadCountKV
		)
		if a.SSMConvolutionKernel > 0 {
			embedKeyGQA += uint64(a.SSMConvolutionKernel - 1*a.SSMInnerSize)
			embedValGQA += uint64(a.SSMStateSize * a.SSMInnerSize)
		}

		krs := kt.RowSizeOf([]uint64{embedKeyGQA * kvSize})
		vrs := vt.RowSizeOf([]uint64{embedValGQA * kvSize})

		ge.KVCache.Key = GGUFBytesScalar(krs * a.BlockCount)
		ge.KVCache.Value = GGUFBytesScalar(vrs * a.BlockCount)
	}

	// Others.
	{
		// Overhead
		ge.Others += GGUFBytesScalar(15 * 1024 * 1024) // NB(thxCode): Magic here.

		// GGML context,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L5015-L5036.
		ggmlCtx := 2 /* buffer count */ * GGMLTensorOverhead() * (uint64(len(gf.TensorInfos)) + 1 + a.BlockCount*3)
		ge.Others += GGUFBytesScalar(ggmlCtx)

		// Output buffer,
		// see https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L11940-L12003.
		outBuffer := 4 /* float32 size */ * (a.VocabularyLength + a.EmbeddingLength) * uint64(ptr.Deref(o.ParallelSize, 1))
		ge.Others += GGUFBytesScalar(outBuffer)
	}

	// Computation graph.
	{
		graphOverhead := GGMLTensorOverhead()*GGMLComputationGraphNodesMaximum +
			GGMLComputationGraphOverhead(GGMLComputationGraphNodesMaximum, false)
		ge.ComputationGraphOverhead += GGUFBytesScalar(graphOverhead)

		var (
			nBatch = min(contextSize, uint64(ptr.Deref(o.BatchSize, 512)))

			inpTokens = GGMLTypeI32.RowSizeOf([]uint64{nBatch})                    // I32 [n_batch]
			inpEmbd   = GGMLTypeF32.RowSizeOf([]uint64{a.EmbeddingLength, nBatch}) // F32 [n_embd, n_batch]
			inpPos    = GGMLTypeI32.RowSizeOf([]uint64{contextSize})               // I32 [n_tokens]
			inpOutIds = GGMLTypeI32.RowSizeOf([]uint64{contextSize})               // I32 [n_output],
			inpKQMask = GGMLTypeF32.RowSizeOf([]uint64{contextSize, nBatch})       // F32 [n_kv, n_batch]
		)
		ge.ComputationGraphOverhead += GGUFBytesScalar(inpTokens + inpEmbd + inpPos + inpKQMask + inpOutIds)
	}

	return ge
}

func (e GGUFEstimate) Sum() GGUFBytesScalar {
	return e.KVCache.Sum() + e.ComputationGraphOverhead + e.Others
}

func (c GGUFKVCacheUsage) Sum() GGUFBytesScalar {
	return c.Key + c.Value
}
