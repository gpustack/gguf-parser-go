package gguf_parser

import (
	"regexp"
	"strings"

	"github.com/thxcode/gguf-parser-go/util/ptr"
)

// Types for LLaMACpp estimation.
type (
	// LLaMACppUsageEstimate represents the estimated result of loading the GGUF file in llama.cpp.
	LLaMACppUsageEstimate struct {
		// FullOffload is the flag to indicate whether the layers are fully offloaded,
		// false for partial offloaded or zero offloaded.
		FullOffload bool `json:"fullOffload"`
		// NoMMap is the flag to indicate whether the file must be loaded without mmap,
		// true for total loaded.
		NoMMap bool `json:"noMMap"`
		// Load is the memory usage for running the GGUF file in RAM.
		Load LLaMACppMemoryUsage `json:"load"`
		// Offload is the memory usage for loading the GGUF file in VRAM.
		Offload LLaMACppMemoryUsage `json:"offload"`
	}

	// LLaMACppMemoryUsage represents the memory usage for expanding the GGUF file in llama.cpp.
	LLaMACppMemoryUsage struct {
		// Footprint is the memory footprint for bootstrapping.
		Footprint GGUFBytesScalar `json:"footprint"`
		// Weight is the memory usage of loading weights.
		Weight LLaMACppWeightUsage `json:"weight"`
		// KVCache is the memory usage of caching previous KV.
		KVCache LLaMACppKVCacheUsage `json:"kvCache"`
		// Computation is the memory usage of computation.
		Computation LLaMACppComputationUsage `json:"computation"`
	}

	// LLaMACppWeightUsage represents the memory usage of loading weights in llama.cpp.
	LLaMACppWeightUsage struct {
		// Input is the memory usage for loading input tensors.
		Input GGUFBytesScalar `json:"input"`
		// Compute is the memory usage for loading compute tensors.
		Compute GGUFBytesScalar `json:"compute"`
		// Output is the memory usage for loading output tensors.
		Output GGUFBytesScalar `json:"output"`
	}

	// LLaMACppKVCacheUsage represents the memory usage of caching previous KV in llama.cpp.
	LLaMACppKVCacheUsage struct {
		// Key is the memory usage for caching previous keys.
		Key GGUFBytesScalar `json:"key"`
		// Value is the memory usage for caching previous values.
		Value GGUFBytesScalar `json:"value"`
	}

	// LLaMACppComputationUsage represents the memory usage of computation in llama.cpp.
	LLaMACppComputationUsage struct {
		// Footprint is the memory footprint for computation.
		Footprint GGUFBytesScalar `json:"footprint"`
		// Input is the memory usage for input.
		Input GGUFBytesScalar `json:"input"`
		// Compute is the memory usage for computation.
		Compute GGUFBytesScalar `json:"graph"`
		// Output is the memory usage for output.
		Output GGUFBytesScalar `json:"output"`
	}
)

// EstimateLLaMACppUsage returns the inference memory usage estimated result of the GGUF file.
func (gf *GGUFFile) EstimateLLaMACppUsage(opts ...LLaMACppUsageEstimateOption) (e LLaMACppUsageEstimate) {
	var o _LLaMACppUsageEstimateOptions
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
	e.FullOffload = a.BlockCount == nOffloadLayers

	// Footprint.
	{
		// Bootstrap.
		e.Load.Footprint = GGUFBytesScalar(10 * 1024 * 1024)
		e.Load.Footprint += gf.Size - gf.ModelSize

		// Tokens.
		fp := t.TokensLength * (4 /* token type */ + 4 /* token score*/)
		if t.Model == "gpt2" {
			fp += t.MergesLength * (48 /* key type */ + 56 /* value type */)
		}
		fp += t.TokensLength * (32 /* id to token vector */ + (24 + 32) /* token to id map*/)

		// Output buffer,
		// see https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L11940-L12003.
		ob := 4 /* float32 size */ * (a.VocabularyLength + a.EmbeddingLength) * nParallel

		e.Load.Footprint += GGUFBytesScalar(fp + ob)
	}

	ls := gf.Layers()
	ioLs, tfLs, _ := ls.Cut([]string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
		"output_norm.bias",
	})
	ipLs, opLs, _ := ioLs.Cut([]string{
		"token_embd.weight",
	})

	// Weight.
	{
		// Compute.
		for i, offloadStart := uint64(0), uint64(len(tfLs))-nOffloadLayers; i < uint64(len(tfLs)); i++ {
			switch {
			case i < nLoadLayers:
				e.Load.Weight.Compute += GGUFBytesScalar(tfLs[i].Bytes())
			case i >= offloadStart:
				e.Offload.Weight.Compute += GGUFBytesScalar(tfLs[i].Bytes())
			}
		}

		// IO,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L4930-L5002.
		e.Load.Weight.Input = GGUFBytesScalar(ipLs.Bytes())
		e.Load.Weight.Output = GGUFBytesScalar(opLs.Bytes())
		if nOffloadLayers == a.BlockCount {
			// Transfer the output weight to VRAM when all layers are offloaded.
			e.Offload.Weight.Output = e.Load.Weight.Output
			e.Load.Weight.Output = 0
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

		embedKeyGQA, embedValGQA := a.EmbeddingKeyGQA, a.EmbeddingValueGQA
		if a.SSMConvolutionKernel > 0 {
			embedKeyGQA += uint64(a.SSMConvolutionKernel - 1*a.SSMInnerSize)
			embedValGQA += uint64(a.SSMStateSize * a.SSMInnerSize)
		}

		krs := kt.RowSizeOf([]uint64{embedKeyGQA * nKV})
		vrs := vt.RowSizeOf([]uint64{embedValGQA * nKV})

		e.Load.KVCache.Key = GGUFBytesScalar(krs * nLoadLayers)
		e.Load.KVCache.Value = GGUFBytesScalar(vrs * nLoadLayers)
		e.Offload.KVCache.Key = GGUFBytesScalar(krs * nOffloadLayers)
		e.Offload.KVCache.Value = GGUFBytesScalar(vrs * nOffloadLayers)
	}

	// Computation.
	{
		// GGML context,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L5015-L5036.
		gc := 2 /* buffer count */ * GGMLTensorOverhead() * (uint64(len(gf.TensorInfos)) + 1 + a.BlockCount*3)

		// Graph overhead.
		oh := GGMLTensorOverhead()*GGMLComputationGraphNodesMaximum +
			GGMLComputationGraphOverhead(GGMLComputationGraphNodesMaximum, false)

		e.Load.Computation.Footprint = GGUFBytesScalar(gc + oh)

		// Tensor usage,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L16149.
		//
		// Firstly, get the usage of input layer.
		var (
			inpTokens = GGMLTypeI32.RowSizeOf([]uint64{nBatch})                    // I32 [n_batch]
			inpEmbd   = GGMLTypeF32.RowSizeOf([]uint64{a.EmbeddingLength, nBatch}) // F32 [n_embd, n_batch]
			inpPos    = GGMLTypeI32.RowSizeOf([]uint64{nContext})                  // I32 [n_tokens]
			inpOutIds = GGMLTypeI32.RowSizeOf([]uint64{nContext})                  // I32 [n_output],
			inpKQMask = GGMLTypeF32.RowSizeOf([]uint64{nContext, nBatch})          // F32 [n_kv, n_batch]
		)
		e.Load.Computation.Input = GGUFBytesScalar(inpTokens + inpEmbd + inpPos + inpKQMask + inpOutIds)
		e.Offload.Computation.Input = GGUFBytesScalar(inpEmbd + inpPos + inpKQMask + inpOutIds)
		// Since the steps between transformer layers are serial,
		// the allocated memory can be reused for the next layer.
		// So, we only consider the usage of the largest layer,
		// which is the last layer by default.
		{
			kvcInc := uint64(e.Load.KVCache.Key + e.Offload.KVCache.Key)
			for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.attn_(norm|q|qkv)\.weight`)) {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
				kvcInc += rs
				switch {
				default:
					continue
				case strings.HasSuffix(l.Name, ".attn_q.weight"):
				case strings.HasSuffix(l.Name, ".attn_qkv.weight"):
					rs = GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[0], nBatch})
				}
				kvcInc += rs * 2 // for RoPE
			}
			ffnInc := uint64(0)
			for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.(attn_norm|ffn_norm|ffn_gate|ffn_up)\.weight`)) {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
				ffnInc += rs
			}
			e.Offload.Computation.Compute = GGUFBytesScalar(max(kvcInc, ffnInc))
			switch {
			case nLoadLayers == 0: // Zero offloaded.
				e.Load.Computation.Compute = GGUFBytesScalar(max(kvcInc, ffnInc))
			case nLoadLayers > 0 && nOffloadLayers > 0: // Partial offloaded.
				ffnInc = 0
				for _, l := range tfLs[nLoadLayers-1].Search(regexp.MustCompile(`.*\.\d+\.ffn_(norm|gate|up)\.weight`)) {
					rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
					ffnInc += rs
				}
				e.Load.Computation.Compute = GGUFBytesScalar(max(kvcInc, ffnInc))
			}
			// Special case: we cannot use mmap for splitting expert weights in MoE.
			if a.ExpertCount > 0 {
				e.NoMMap = len(tfLs[0].Search(regexp.MustCompile(`.*\.\d+\.ffn_gate_exps\.weight`))) == 0
			}
		}
		// Finally, get the usage of output layer.
		{
			outInc := inpEmbd
			if l, ok := opLs.Get("output.weight"); ok {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
				outInc += rs
			}
			e.Offload.Computation.Output = GGUFBytesScalar(outInc)
		}
	}

	return e
}

// LLaMACppUsageEstimateSummery represents the summary of the usage for loading the GGUF file in llama.cpp.
type LLaMACppUsageEstimateSummery struct {
	// UMA represents the usage of Unified Memory Architecture.
	UMA GGUFBytesScalar `json:"uma"`
	// NonUMA represents the usage of Non-Unified Memory Architecture.
	NonUMA struct {
		// Load is the memory usage for loading the GGUF file in Load.
		RAM GGUFBytesScalar `json:"ram"`
		// VRAM is the memory usage for loading the GGUF file in VRAM.
		VRAM GGUFBytesScalar `json:"vram"`
	} `json:"nonUMA"`
}

func (e LLaMACppUsageEstimate) Summarize(mmap bool) (es LLaMACppUsageEstimateSummery) {
	// UMA.
	{
		kv := e.Load.KVCache.Sum() + e.Offload.KVCache.Sum()
		wg := e.Load.Weight.Sum() + e.Offload.Weight.Sum()
		es.UMA = e.Load.Footprint + max(kv, e.Load.Computation.Sum()) + wg
		if !e.NoMMap && mmap {
			es.UMA -= wg
		}
	}

	// TODO(thxCode): complete more cases,
	//  and support optional parameters for the following constants.

	// Footprint,
	// see https://github.com/ggerganov/llama.cpp/blob/f578b86b2123d0f92afbaa98a031df4d4464e582/llama.cpp#L2454-L2486.
	const (
		// The function `cudaMemGetInfo` occupies some memory,
		// see https://github.com/ggerganov/llama.cpp/blob/f578b86b2123d0f92afbaa98a031df4d4464e582/ggml-cuda.cu#L3009-L3013,
		// and https://stackoverflow.com/questions/64854862/free-memory-occupied-by-cudamemgetinfo.
		cudaFootprint = GGUFBytesScalar(150 * 1024 * 1024)
	)

	// NonUMA.
	{
		wg := e.Load.Weight.Sum()
		es.NonUMA.RAM = cudaFootprint + e.Load.Footprint + e.Load.KVCache.Sum() + e.Load.Computation.Sum() + wg - e.Load.Computation.Compute
		if !e.NoMMap && (mmap || e.FullOffload) {
			es.NonUMA.RAM -= wg
		}
		es.NonUMA.VRAM = e.Offload.Footprint + e.Offload.Weight.Sum() + e.Offload.KVCache.Sum() + e.Offload.Computation.Sum()
	}

	return es
}

func (u LLaMACppWeightUsage) Sum() GGUFBytesScalar {
	return u.Input + u.Compute + u.Output
}

func (u LLaMACppKVCacheUsage) Sum() GGUFBytesScalar {
	return u.Key + u.Value
}

func (u LLaMACppComputationUsage) Sum() GGUFBytesScalar {
	r := u.Input + u.Compute
	if r < u.Output {
		r = u.Output
	}
	return u.Footprint + r
}
