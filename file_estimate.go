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
		// Layers is the number of layers for loading the GGUF file.
		Layers uint64 `json:"layers"`
		// OffloadLayers is the number of layers to offload.
		OffloadLayers uint64 `json:"offloadLayers"`
		// RAM is the memory usage for loading the GGUF file in RAM.
		RAM LLaMACppMemoryUsage `json:"ram"`
		// VRAM is the memory usage for loading the GGUF file in VRAM.
		VRAM LLaMACppMemoryUsage `json:"vram"`
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
	e.Layers = a.BlockCount
	e.OffloadLayers = nOffloadLayers

	// Footprint.
	{
		// Bootstrap.
		e.RAM.Footprint = GGUFBytesScalar(5 * 1024 * 1024)

		// Tokens.
		fp := uint64(t.TokensSize)
		fp += t.TokensLength * (4 /* token type */ + 4 /* token score*/)
		if t.Model == "gpt2" {
			fp += uint64(t.MergesSize)
			fp += t.MergesLength * (48 /* key type */ + 56 /* value type */)
		}

		// Output buffer,
		// see https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L11940-L12003.
		ob := 4 /* float32 size */ * (a.VocabularyLength + a.EmbeddingLength) * nParallel

		e.RAM.Footprint += GGUFBytesScalar(fp + ob)
	}

	ls := gf.Layers()
	ioLs, tfLs, _ := ls.Cut([]string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
	})
	ipLs, opLs, _ := ioLs.Cut([]string{
		"token_embd.weight",
	})

	// Weight.
	{
		// Compute.
		for i, offloadStart := uint64(0), uint64(len(tfLs))-nOffloadLayers; i < uint64(len(tfLs)); i++ {
			e.RAM.Weight.Compute += GGUFBytesScalar(tfLs[i].Bytes())
			if i >= offloadStart {
				e.VRAM.Weight.Compute += GGUFBytesScalar(tfLs[i].Bytes())
			}
		}

		// IO,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L4930-L5002.
		e.RAM.Weight.Input = GGUFBytesScalar(ipLs.Bytes())
		e.RAM.Weight.Output = GGUFBytesScalar(opLs.Bytes())
		if nOffloadLayers == a.BlockCount {
			// Transfer the output weight to VRAM when all layers are offloaded.
			e.VRAM.Weight.Output = e.RAM.Weight.Output
			e.RAM.Weight.Output = 0
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

		e.RAM.KVCache.Key = GGUFBytesScalar(krs * nLoadLayers)
		e.RAM.KVCache.Value = GGUFBytesScalar(vrs * nLoadLayers)
		e.VRAM.KVCache.Key = GGUFBytesScalar(krs * nOffloadLayers)
		e.VRAM.KVCache.Value = GGUFBytesScalar(vrs * nOffloadLayers)
	}

	// Computation.
	{
		// GGML context,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L5015-L5036.
		gc := 2 /* buffer count */ * GGMLTensorOverhead() * (uint64(len(gf.TensorInfos)) + 1 + a.BlockCount*3)

		// Graph overhead.
		oh := GGMLTensorOverhead()*GGMLComputationGraphNodesMaximum +
			GGMLComputationGraphOverhead(GGMLComputationGraphNodesMaximum, false)

		e.RAM.Computation.Footprint = GGUFBytesScalar(gc + oh)

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
		e.RAM.Computation.Input = GGUFBytesScalar(inpTokens + inpEmbd + inpPos + inpKQMask + inpOutIds)
		e.VRAM.Computation.Input = GGUFBytesScalar(inpEmbd + inpPos + inpKQMask + inpOutIds)
		// Since the steps between transformer layers are serial,
		// the allocated memory can be reused for the next layer.
		// So, we only consider the usage of the largest layer,
		// which is the last layer by default.
		{
			kvcInc := uint64(e.RAM.KVCache.Key + e.VRAM.KVCache.Key)
			for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.attn_(norm|q)\.weight`)) {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
				kvcInc += rs
				if strings.HasSuffix(l.Name, ".attn_q.weight") {
					kvcInc += rs // for RoPE
				}
			}
			ffnInc := uint64(0)
			for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.(attn_norm|ffn_norm|ffn_gate|ffn_up)\.weight`)) {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
				ffnInc += rs
			}
			e.VRAM.Computation.Compute = GGUFBytesScalar(max(kvcInc, ffnInc))
			if nLoadLayers > 0 {
				ffnInc = 0
				for _, l := range tfLs[nLoadLayers-1].Search(regexp.MustCompile(`.*\.\d+\.ffn_(norm|gate|up)\.weight`)) {
					rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
					ffnInc += rs
				}
				e.RAM.Computation.Compute = GGUFBytesScalar(max(kvcInc, ffnInc))
			}
		}
		// Finally, get the usage of output layer.
		{
			outInc := inpEmbd
			if l, ok := opLs.Get("output.weight"); ok {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nBatch})
				outInc += rs
			}
			e.VRAM.Computation.Output = GGUFBytesScalar(outInc)
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
		// RAM is the memory usage for loading the GGUF file in RAM.
		RAM GGUFBytesScalar `json:"ram"`
		// VRAM is the memory usage for loading the GGUF file in VRAM.
		VRAM GGUFBytesScalar `json:"vram"`
	} `json:"nonUMA"`
}

func (e LLaMACppUsageEstimate) Summarize(mmap bool) (es LLaMACppUsageEstimateSummery) {
	// UMA.
	{
		es.UMA = e.RAM.Footprint
		switch kv := e.RAM.KVCache.Sum() + e.VRAM.KVCache.Sum(); {
		case e.OffloadLayers == 0:
			cp := e.RAM.Computation.Sum()
			es.UMA += max(kv, cp)
		case e.Layers == e.OffloadLayers:
			cp := e.VRAM.Computation.Sum()
			es.UMA += max(kv, cp)
		default:
			es.UMA += max(kv, max(e.RAM.Computation.Sum(), e.VRAM.Computation.Sum()))
		}
		if !mmap {
			es.UMA += e.RAM.Weight.Sum()
		}
	}

	// NonUMA.
	{
		es.NonUMA.RAM = e.RAM.Footprint + e.RAM.KVCache.Sum() + e.RAM.Computation.Sum()
		if !mmap && e.Layers != e.OffloadLayers {
			es.NonUMA.RAM += e.RAM.Weight.Sum()
		}
		es.NonUMA.VRAM = e.VRAM.Footprint + e.VRAM.Weight.Sum() + e.VRAM.KVCache.Sum() + e.VRAM.Computation.Sum()
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
