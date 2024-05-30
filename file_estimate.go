package gguf_parser

// GGUFEstimate represents the estimated result of the GGUF file.
type GGUFEstimate struct {
	// Offload is the offloaded layers usage.
	Offload *GGUFMemoryUsage `json:"offload,omitempty"`
	// Total is the total memory usage.
	Total GGUFMemoryUsage `json:"total"`
}

type (
	// GGUFMemoryUsage represents the memory usage of the GGUF file.
	GGUFMemoryUsage struct {
		// KVCache is the usage of key-value cache.
		KVCache GGUFKVCacheUsage `json:"kvCache"`
		// Compute is the usage of transformer layers.
		Compute GGUFBytesScalar `json:"compute"`
		// IO is the usage of input/output layers.
		IO GGUFBytesScalar `json:"io"`
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

	ge.Offload, ge.Total = gf.estimateMemoryUsage(gf.Architecture(), o)
	return ge
}

func (m GGUFMemoryUsage) Sum() GGUFBytesScalar {
	return m.Compute + m.KVCache.Sum() + m.IO
}

func (c GGUFKVCacheUsage) Sum() GGUFBytesScalar {
	return c.Key + c.Value
}

func (gf *GGUFFile) estimateMemoryUsage(a GGUFArchitectureMetadata, o _GGUFEstimateOptions) (offload *GGUFMemoryUsage, total GGUFMemoryUsage) {
	if o.OffloadLayers != nil {
		offload = &GGUFMemoryUsage{}
	}

	// KV cache.
	// https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L2479-L2501
	{
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
			kvSize      = a.MaximumContextLength
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

		krs := kt.RowSizeOf([]uint64{embedKeyGQA * kvSize})
		vrs := vt.RowSizeOf([]uint64{embedValGQA * kvSize})

		if offload != nil {
			v := *o.OffloadLayers
			if v > a.BlockCount {
				v = a.BlockCount
			}
			offload.KVCache.Key = GGUFBytesScalar(krs * v)
			offload.KVCache.Value = GGUFBytesScalar(vrs * v)
		}

		total.KVCache.Key = GGUFBytesScalar(krs * a.BlockCount)
		total.KVCache.Value = GGUFBytesScalar(vrs * a.BlockCount)
	}

	ls := gf.Layers()
	bls, als, _ := ls.Cut([]string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
	})

	// IO.
	total.IO = GGUFBytesScalar(bls.Bytes())

	// Compute.
	if offload != nil {
		v := *o.OffloadLayers
		if v >= a.BlockCount {
			offload.Compute = GGUFBytesScalar(als.Bytes())
		} else {
			for i := uint64(len(als) - 1); i >= uint64(len(als))-v; i-- {
				offload.Compute += GGUFBytesScalar(als[i].Bytes())
			}
		}
	}
	total.Compute = GGUFBytesScalar(als.Bytes())

	return offload, total
}
