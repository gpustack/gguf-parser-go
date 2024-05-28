package gguf_parser

// GGUFEstimate represents the estimated result of the GGUF file.
type GGUFEstimate struct {
	// MemoryTotal is the total memory usage.
	MemoryTotal GGUFBytesScalar `json:"memoryTotal"`
	// MemoryLoad is memory usage to load the model.
	MemoryLoad GGUFBytesScalar `json:"memoryLoad"`
	// KVCache is the usage of key-value cache.
	KVCache GGUFEstimateKVCache `json:"kvCache"`
}

// GGUFEstimateKVCache represents the usage of kv-cache.
type GGUFEstimateKVCache struct {
	// MemoryTotal is the total memory usage.
	MemoryTotal GGUFBytesScalar `json:"memoryTotal"`
	// MemoryKey is the memory usage of the cached key.
	MemoryKey GGUFBytesScalar `json:"memoryKey"`
	// MemoryValue is the memory usage of the cached value.
	MemoryValue GGUFBytesScalar `json:"memoryValue"`
}

// Estimate returns the estimated result of the GGUF file.
func (gf *GGUFFile) Estimate(opts ...GGUFEstimateOption) (ge GGUFEstimate) {
	var o _GGUFEstimateOptions
	for _, opt := range opts {
		opt(&o)
	}

	ge.MemoryLoad = gf.ModelSize
	ge.KVCache = gf.estimateKVCache(gf.Architecture(), o)
	ge.MemoryTotal = ge.MemoryLoad + ge.KVCache.MemoryTotal

	return ge
}

// estimateKVCache estimates the key-value cache,
// which is inspired by https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L2479-L2501
func (gf *GGUFFile) estimateKVCache(a GGUFArchitectureMetadata, o _GGUFEstimateOptions) (kv GGUFEstimateKVCache) {
	kt, vt := GGMLTypeF16, GGMLTypeF16

	if o.CacheKeyType != nil {
		kt = *o.CacheKeyType
	}
	if o.CacheValueType != nil {
		vt = *o.CacheValueType
	}

	var (
		embedKeyGQA = uint64(a.AttentionKeyLength) * a.AttentionHeadCountKV
		embedValGQA = uint64(a.AttentionValueLength) * a.AttentionHeadCountKV
		kvSize      = a.ContextLength
	)
	{
		// Correct.
		if a.SSMConvolutionKernel > 0 {
			embedKeyGQA += uint64(a.SSMConvolutionKernel - 1*a.SSMInnerSize)
			embedValGQA += uint64(a.SSMStateSize * a.SSMInnerSize)
		}
		if o.ContextSize != nil {
			kvSize = uint64(*o.ContextSize)
		}
	}

	kv.MemoryKey = GGUFBytesScalar(kt.RowSizeOf([]uint64{embedKeyGQA * kvSize}) * a.BlockCount)
	kv.MemoryValue = GGUFBytesScalar(vt.RowSizeOf([]uint64{embedValGQA * kvSize}) * a.BlockCount)
	kv.MemoryTotal = kv.MemoryKey + kv.MemoryValue

	return kv
}
