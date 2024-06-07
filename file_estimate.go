package gguf_parser

import (
	"regexp"
	"strings"

	"github.com/thxcode/gguf-parser-go/util/ptr"
)

// GGUFEstimate represents the estimated result of the GGUF file.
type GGUFEstimate struct {
	// Load is the memory usage of the load part.
	Load GGUFMemoryUsage `json:"load"`
	// Offload is the memory usage of the offload part.
	Offload GGUFMemoryUsage `json:"offload"`
}

type (
	// GGUFMemoryUsage represents the memory usage of the GGUF file.
	GGUFMemoryUsage struct {
		// Weight is the memory usage of weight.
		Weight GGUFWeightUsage `json:"weight"`
		// KVCache is the usage of key-value cache.
		KVCache GGUFKVCacheUsage `json:"kvCache"`
		// Tokens is the memory usage of token.
		Tokens GGUFBytesScalar `json:"tokens"`
		// Compute is the memory usage of computation.
		Compute GGUFComputeUsage `json:"compute"`
	}

	// GGUFWeightUsage represents the memory usage of model weight.
	GGUFWeightUsage struct {
		// Compute is the memory usage of computing.
		Compute GGUFBytesScalar `json:"compute"`
		// Input is the memory usage of input.
		Input GGUFBytesScalar `json:"input"`
		// Output is the memory usage of output.
		Output GGUFBytesScalar `json:"output"`
	}

	// GGUFKVCacheUsage represents the usage of kv-cache.
	GGUFKVCacheUsage struct {
		// Key is the memory usage of the cached key.
		Key GGUFBytesScalar `json:"key"`
		// Value is the memory usage of the cached value.
		Value GGUFBytesScalar `json:"value"`
	}

	// GGUFComputeUsage represents the memory usage of computation.
	GGUFComputeUsage struct {
		// Graph is the memory usage of computation graph.
		Graph GGUFBytesScalar `json:"graph"`
		// Others is the trivial usage.
		Others GGUFBytesScalar `json:"others"`
	}
)

// Estimate returns the inference usage estimated result of the GGUF file.
func (gf *GGUFFile) Estimate(opts ...GGUFEstimateOption) (ge GGUFEstimate) {
	var o _GGUFEstimateOptions
	for _, opt := range opts {
		opt(&o)
	}

	a, t := gf.Architecture(), gf.Tokenizer()

	nContext := a.MaximumContextLength
	if o.ContextSize != nil {
		nContext = uint64(*o.ContextSize)
	}

	var (
		nLoadLayers    = a.BlockCount
		nOffloadLayers uint64
		nBatch         = min(nContext, uint64(ptr.Deref(o.BatchSize, 512)))
		nParallel      = uint64(ptr.Deref(o.ParallelSize, 1))
	)
	{
		if v := o.OffloadLayers; v == nil {
			o.OffloadLayers = ptr.To(a.BlockCount)
			nOffloadLayers = nLoadLayers
		} else if *v > 0 {
			nOffloadLayers = *v
			if nOffloadLayers > nLoadLayers {
				nOffloadLayers = nLoadLayers
			}
		}
		nLoadLayers -= nOffloadLayers
	}

	ls := gf.Layers()
	ioLs, tfLs, _ := ls.Cut([]string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
	})

	// Model weight.
	{
		// Compute.
		for i, offloadStart := uint64(0), uint64(len(tfLs))-nOffloadLayers; i < uint64(len(tfLs)); i++ {
			switch {
			case i < nLoadLayers:
				ge.Load.Weight.Compute += GGUFBytesScalar(tfLs[i].Bytes())
			case i >= offloadStart:
				ge.Offload.Weight.Compute += GGUFBytesScalar(tfLs[i].Bytes())
			}
		}

		// IO,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L4930-L5002.
		inpLs, outLs, _ := ioLs.Cut([]string{
			"token_embd.weight",
		})
		ge.Load.Weight.Input = GGUFBytesScalar(inpLs.Bytes())
		ge.Load.Weight.Output = GGUFBytesScalar(outLs.Bytes())
		if nOffloadLayers == a.BlockCount {
			ge.Offload.Weight.Output = ge.Load.Weight.Output
			ge.Load.Weight.Output = 0
		}
	}

	// KV cache,
	// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L2479-L2501.
	{
		kt, vt := GGMLTypeF16, GGMLTypeF16
		nKV := nContext
		if o.CacheKeyType != nil {
			kt = *o.CacheKeyType
		}
		if o.CacheValueType != nil {
			vt = *o.CacheValueType
		}
		if a.Architecture == "mamba" {
			// See https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L16122-L16129.
			kt, vt = GGMLTypeF32, GGMLTypeF32
			nKV = nParallel
		}

		var (
			embedKeyGQA = uint64(a.AttentionKeyLength) * a.AttentionHeadCountKV
			embedValGQA = uint64(a.AttentionValueLength) * a.AttentionHeadCountKV
		)
		if a.SSMConvolutionKernel > 0 {
			embedKeyGQA += uint64(a.SSMConvolutionKernel - 1*a.SSMInnerSize)
			embedValGQA += uint64(a.SSMStateSize * a.SSMInnerSize)
		}

		krs := kt.RowSizeOf([]uint64{embedKeyGQA * nKV})
		vrs := vt.RowSizeOf([]uint64{embedValGQA * nKV})

		ge.Load.KVCache.Key = GGUFBytesScalar(krs * nLoadLayers)
		ge.Load.KVCache.Value = GGUFBytesScalar(vrs * nLoadLayers)
		ge.Offload.KVCache.Key = GGUFBytesScalar(krs * nOffloadLayers)
		ge.Offload.KVCache.Value = GGUFBytesScalar(vrs * nOffloadLayers)
	}

	// Tokens.
	ge.Load.Tokens += GGUFBytesScalar(t.TokensSize)
	ge.Load.Tokens += GGUFBytesScalar(t.TokensLength * (4 /* token type */ + 4 /* token score*/))
	if t.Model == "gpt2" {
		ge.Load.Tokens += GGUFBytesScalar(t.MergesSize)
		ge.Load.Tokens += GGUFBytesScalar(t.MergesLength * (48 /* key type */ + 56 /* value type */))
	}

	// Compute.
	{
		// Bootstrap.
		ge.Load.Compute.Others += GGUFBytesScalar(15 * 1024 * 1024)

		// GGML context,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L5015-L5036.
		ggmlCtx := 2 /* buffer count */ * GGMLTensorOverhead() * (uint64(len(gf.TensorInfos)) + 1 + a.BlockCount*3)
		ge.Load.Compute.Others += GGUFBytesScalar(ggmlCtx)

		// Output buffer,
		// see https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L11940-L12003.
		outBuffer := 4 /* float32 size */ * (a.VocabularyLength + a.EmbeddingLength) * nParallel
		ge.Load.Compute.Others += GGUFBytesScalar(outBuffer)

		// Graph overhead.
		graphOverhead := GGMLTensorOverhead()*GGMLComputationGraphNodesMaximum +
			GGMLComputationGraphOverhead(GGMLComputationGraphNodesMaximum, false)
		ge.Load.Compute.Others += GGUFBytesScalar(graphOverhead)
	}

	// Computation graph.
	{
		// Tensor usage,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L16149.
		//
		// Firstly, get the usage of input tensors.
		var (
			inpTokens = GGMLTypeI32.RowSizeOf([]uint64{nBatch})                    // I32 [n_batch]
			inpEmbd   = GGMLTypeF32.RowSizeOf([]uint64{a.EmbeddingLength, nBatch}) // F32 [n_embd, n_batch]
			inpPos    = GGMLTypeI32.RowSizeOf([]uint64{nContext})                  // I32 [n_tokens]
			inpOutIds = GGMLTypeI32.RowSizeOf([]uint64{nContext})                  // I32 [n_output],
			inpKQMask = GGMLTypeF32.RowSizeOf([]uint64{nContext, nBatch})          // F32 [n_kv, n_batch]
		)
		ge.Load.Compute.Graph += GGUFBytesScalar(inpTokens + inpEmbd + inpPos + inpKQMask + inpOutIds)
		if nOffloadLayers > 0 {
			ge.Offload.Compute.Graph += GGUFBytesScalar(inpEmbd + inpPos + inpKQMask + inpOutIds)
		}
		// Since the steps between transformer layers are serial,
		// the allocated memory can be reused for the next layer.
		// So, we only consider the usage of the largest layer,
		// which is the last layer by default.
		kvcInc := uint64(ge.Load.KVCache.Key + ge.Offload.KVCache.Key)
		for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.attn_(norm|q)\.weight`)) {
			rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
			kvcInc += rs
			if strings.HasSuffix(l.Name, ".attn_q.weight") {
				kvcInc += rs // for RoPE
			}
		}
		var ffnInc uint64
		for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.(attn_norm|ffn_norm|ffn_gate|ffn_up)\.weight`)) {
			rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
			ffnInc += rs
		}
		if nLoadLayers == a.BlockCount {
			ge.Load.Compute.Graph += GGUFBytesScalar(max(kvcInc, ffnInc))
		} else {
			ge.Offload.Compute.Graph += GGUFBytesScalar(max(kvcInc, ffnInc))
			if nLoadLayers > 0 {
				ffnInc = 0
				for _, l := range tfLs[nLoadLayers-1].Search(regexp.MustCompile(`.*\.\d+\.ffn_(norm|gate|up)\.weight`)) {
					rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
					ffnInc += rs
				}
				ge.Load.Compute.Graph += GGUFBytesScalar(max(kvcInc, ffnInc))
			}
		}
	}

	return ge
}

type (
	GGUFEstimateSum struct {
		// UMA is the usage of unified memory architecture.
		UMA GGUFEstimateSumItem `json:"uma"`
		// NonUMA is the usage of non-unified memory architecture.
		NonUMA GGUFEstimateSumItem `json:"nonUMA"`
	}
	GGUFEstimateSumItem struct {
		// RAM is the memory usage of the RAM.
		RAM GGUFBytesScalar `json:"ram"`
		// VRAM is the memory usage of the VRAM.
		VRAM GGUFBytesScalar `json:"vram"`
	}
)

func (e GGUFEstimate) Sum(mmap bool) (gs GGUFEstimateSum) {
	gs.UMA = GGUFEstimateSumItem{
		RAM:  e.Load.KVCache.Sum() + e.Offload.KVCache.Sum() + e.Load.Tokens + e.Load.Compute.Others,
		VRAM: e.Offload.Compute.Sum(),
	}
	if !mmap {
		gs.UMA.RAM += e.Load.Weight.Sum()
		gs.UMA.VRAM += e.Offload.Weight.Sum()
	}
	gs.NonUMA = GGUFEstimateSumItem{
		RAM:  e.Load.KVCache.Sum() + e.Load.Tokens + e.Load.Compute.Sum(),
		VRAM: e.Offload.KVCache.Sum() + e.Offload.Compute.Sum(),
	}
	if !mmap {
		gs.NonUMA.RAM += e.Load.Weight.Sum()
		gs.NonUMA.VRAM += e.Offload.Weight.Sum()
	}
	return gs
}

func (w GGUFWeightUsage) Sum() GGUFBytesScalar {
	return w.Compute + w.Input + w.Output
}

func (c GGUFKVCacheUsage) Sum() GGUFBytesScalar {
	return c.Key + c.Value
}

func (c GGUFComputeUsage) Sum() GGUFBytesScalar {
	return c.Graph + c.Others
}
