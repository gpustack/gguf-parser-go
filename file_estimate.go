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
		// Architecture describes what architecture this model implements.
		Architecture string `json:"architecture"`
		// FlashAttention is the flag to indicate whether enable the flash attention,
		// true for enable.
		FlashAttention bool `json:"flashAttention"`
		// ContextSize is the size of the context.
		ContextSize uint64 `json:"contextSize"`
		// OffloadLayers is the number of offloaded layers.
		OffloadLayers uint64 `json:"offloadLayers"`
		// FullOffloaded is the flag to indicate whether the layers are fully offloaded,
		// false for partial offloaded or zero offloaded.
		FullOffloaded bool `json:"fullOffloaded"`
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
	if o.CacheKeyType == nil {
		o.CacheKeyType = ptr.To(GGMLTypeF16)
	}
	if o.CacheValueType == nil {
		o.CacheValueType = ptr.To(GGMLTypeF16)
	}
	if o.OffloadKVCache == nil {
		o.OffloadKVCache = ptr.To(true)
	}
	if o.PhysicalBatchSize == nil {
		o.PhysicalBatchSize = ptr.To(int32(512))
	}

	// Architecture and tokenizer metadata.
	var (
		a GGUFArchitectureMetadata
		t GGUFTokenizerMetadata
	)
	if o.Architecture != nil {
		a = *o.Architecture
	} else {
		a = gf.Architecture()
	}
	if o.Tokenizer != nil {
		t = *o.Tokenizer
	} else {
		t = gf.Tokenizer()
	}
	e.Architecture = a.Architecture

	// Flash attention.
	{
		// Quantization requires flash attention,
		// see https://github.com/ggerganov/llama.cpp/blob/172c8256840ffd882ab9992ecedbb587d9b21f15/llama.cpp#L16055-L16058.
		if *o.CacheValueType > GGMLTypeF16 && !o.FlashAttention {
			o.FlashAttention = true
		}
		// Grok is not compatible with flash attention,
		// see https://github.com/ggerganov/llama.cpp/blob/172c8256840ffd882ab9992ecedbb587d9b21f15/llama.cpp#L16050-L16053.
		if a.Architecture == "grok" {
			o.FlashAttention = false
		}

		e.FlashAttention = o.FlashAttention
	}

	// Init hyperparameters,
	// https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L6957-L7000.
	var (
		nContext  uint64
		nTokens   uint64
		nBatch    uint64
		nOutputs  uint64
		nParallel uint64
		nKV       uint64
	)
	{
		nContext = a.MaximumContextLength
		if o.ContextSize != nil {
			nContext = uint64(*o.ContextSize)
		}
		// Correct token size,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L12221-L12224.
		nTokens = min(nContext, uint64(*o.PhysicalBatchSize))
		nBatch = nTokens
		nOutputs = nTokens
		nParallel = uint64(ptr.Deref(o.ParallelSize, 1))
		nKV = nContext

		// For mamba,
		// see https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L16122-L16129.
		if a.Architecture == "mamba" {
			nKV = nParallel
			o.CacheKeyType = ptr.To(GGMLTypeF32)
			o.CacheValueType = ptr.To(GGMLTypeF32)
		}

		e.ContextSize = nContext
	}

	// Full offload: isOffloadOutputLayer && nLoadLayers == 0.
	// Partial offload: nLoadLayers > 0 && nOffloadLayers > 0.
	// Zero offload: nOffloadLayers == 0.
	var (
		nLoadLayers          = a.BlockCount
		nOffloadLayers       uint64
		isOffloadOutputLayer bool
	)
	{
		if v := o.OffloadLayers; v == nil {
			o.OffloadLayers = ptr.To(a.BlockCount)
			nOffloadLayers = a.BlockCount
			isOffloadOutputLayer = true
		} else if *v != 0 {
			nOffloadLayers = *v
			if nOffloadLayers > a.BlockCount {
				isOffloadOutputLayer = true
				nOffloadLayers = a.BlockCount
			}
		}
		nLoadLayers -= nOffloadLayers

		e.FullOffloaded = isOffloadOutputLayer && nLoadLayers == 0
		e.OffloadLayers = nOffloadLayers
	}

	// Footprint.
	{
		// Bootstrap.
		e.Load.Footprint = GGUFBytesScalar(5*1024*1024) /* model load */ + (gf.Size - gf.ModelSize) /* metadata */

		// Tokens,
		// https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L6380-L6384.
		fp := t.TokensLength * (4 /* token type */ + 4 /* token score*/)
		if t.Model == "gpt2" {
			fp += t.MergesLength * (48 /* key type */ + 56 /* value type */)
		}
		fp += t.TokensLength * (32 /* id to token vector */ + (24 + 32) /* token to id map*/)
		e.Load.Footprint += GGUFBytesScalar(fp)

		// Output buffer,
		// see https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L11940-L12003.
		ob := 4 /* float32 size */ * (a.VocabularyLength + a.EmbeddingLength) * nParallel
		e.Load.Footprint += GGUFBytesScalar(ob)
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
		if _, ok := opLs.Get("output.weight"); ok {
			e.Load.Weight.Output = GGUFBytesScalar(opLs.Bytes())
		} else {
			e.Load.Weight.Output = GGUFBytesScalar(opLs.Bytes()) + e.Load.Weight.Input /* duplicate the input layer */
		}
	}

	// KV cache,
	// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L2479-L2501.
	{
		krs := o.CacheKeyType.RowSizeOf([]uint64{a.EmbeddingKeyGQA * nKV})
		vrs := o.CacheValueType.RowSizeOf([]uint64{a.EmbeddingValueGQA * nKV})

		e.Load.KVCache.Key = GGUFBytesScalar(krs * nLoadLayers)
		e.Load.KVCache.Value = GGUFBytesScalar(vrs * nLoadLayers)
		e.Offload.KVCache.Key = GGUFBytesScalar(krs * nOffloadLayers)
		e.Offload.KVCache.Value = GGUFBytesScalar(vrs * nOffloadLayers)

		if !*o.OffloadKVCache {
			e.Load.KVCache.Key += e.Offload.KVCache.Key
			e.Load.KVCache.Value += e.Offload.KVCache.Value
			e.Offload.KVCache.Key = GGUFBytesScalar(0)
			e.Offload.KVCache.Value = GGUFBytesScalar(0)
		}
	}

	// Computation.
	{
		// Bootstrap, compute metadata,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L16135-L16136.
		cm := GGMLTensorOverhead()*GGMLComputationGraphNodesMaximum +
			GGMLComputationGraphOverhead(GGMLComputationGraphNodesMaximum, false)
		e.Load.Computation.Footprint = GGUFBytesScalar(cm)

		// Scheduler overhead,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L16149.
		e.Load.Computation.Footprint += GGUFBytesScalar(4 * 1024 * 1024)

		// GGML context,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L5015-L5036.
		gc := 2 /* buffer count */ * GGMLTensorOverhead() * (uint64(len(gf.TensorInfos)) + 1 + a.BlockCount*3)
		e.Load.Computation.Footprint += GGUFBytesScalar(gc)

		// Tensor usage,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L16149.
		//
		// First, get the usage of input layer,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L2279-L2290.
		var (
			inpTokens = GGMLTypeI32.RowSizeOf([]uint64{nBatch})                    // I32 [n_batch]
			inpEmbd   = GGMLTypeF32.RowSizeOf([]uint64{a.EmbeddingLength, nBatch}) // F32 [n_embd, n_batch]
			inpPos    = GGMLTypeI32.RowSizeOf([]uint64{nBatch})                    // I32 [n_batch]
			inpOutIds = GGMLTypeI32.RowSizeOf([]uint64{nOutputs})                  // I32 [n_outputs],
			inpKQMask = GGMLTypeF32.RowSizeOf([]uint64{nKV, nBatch})               // F32 [n_kv, n_batch]
			inpSMask  = GGMLTypeF32.RowSizeOf([]uint64{1, nKV})                    // F32 [1, n_kv]
			inpSSeq   = GGMLTypeI32.RowSizeOf([]uint64{nKV, nBatch})               // I32 [n_kv, n_batch]
		)
		if a.Architecture == "mamba" {
			e.Load.Computation.Input = GGUFBytesScalar(inpTokens + inpEmbd + inpSMask + inpSSeq + inpOutIds)
			e.Offload.Computation.Input = GGUFBytesScalar(inpEmbd + inpSMask + inpSSeq + inpOutIds)
		} else {
			e.Load.Computation.Input = GGUFBytesScalar(inpTokens + inpEmbd + inpPos + inpKQMask + inpOutIds)
			e.Offload.Computation.Input = GGUFBytesScalar(inpEmbd + inpPos + inpKQMask + inpOutIds)
		}
		// Since the steps between transformer layers are serial,
		// the allocated memory can be reused for the next layer.
		// So, we only consider the usage of the largest layer,
		// which is the last layer by default.
		if a.Architecture == "mamba" {
			convInc := GGMLTypeF32.RowSizeOf([]uint64{a.EmbeddingKeyGQA, nKV}) // F32 [n_embd_key_gqa, n_kv] reshape
			for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.(attn_norm|ssm_in|ssm_conv1d)\.weight`)) {
				if !strings.HasSuffix(l.Name, ".ssm_conv1d.weight") {
					rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nTokens})
					convInc += rs
					continue
				}
				// https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L10379.
				rs := GGMLTypeF32.RowSizeOf([]uint64{uint64(a.SSMInnerSize)*nTokens + uint64(a.SSMConvolutionKernel)*uint64(a.SSMInnerSize)*nKV})
				convInc += rs
			}
			ssmInc := uint64(0)
			for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.ssm_(dt\.weight|a)`)) {
				if !strings.HasSuffix(l.Name, ".ssm_a") {
					rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nTokens})
					ssmInc += rs
					continue
				}
				// https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L10413.
				rs := GGMLTypeF32.RowSizeOf([]uint64{uint64(a.SSMInnerSize)*nTokens + uint64(a.SSMStateSize)*uint64(a.SSMInnerSize)*nKV})
				ssmInc += rs
			}
			e.Offload.Computation.Compute = GGUFBytesScalar(convInc + ssmInc)
		} else {
			loadAttnInc, offloadAttnInc := uint64(0), uint64(0)
			if o.FlashAttention {
				// https://github.com/ggerganov/llama.cpp/blob/172c8256840ffd882ab9992ecedbb587d9b21f15/llama.cpp#L7387.
				offloadAttnInc = GGMLTypeF16.RowSizeOf([]uint64{nKV, nTokens})
				for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.attn_(norm|q|qkv)\.weight`)) {
					if strings.HasSuffix(l.Name, ".attn_norm.weight") {
						rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nTokens})
						offloadAttnInc += rs
						continue
					}
					rs := l.Bytes()
					offloadAttnInc += rs
				}
				// https://github.com/ggerganov/llama.cpp/blob/172c8256840ffd882ab9992ecedbb587d9b21f15/llama.cpp#L6986-L6992.
				rs := o.CacheKeyType.RowSizeOf([]uint64{uint64(a.AttentionKeyLength), nKV, a.AttentionHeadCountKV})
				offloadAttnInc += rs
				// https://github.com/ggerganov/llama.cpp/blob/172c8256840ffd882ab9992ecedbb587d9b21f15/llama.cpp#L7000-L7007.
				rs = o.CacheValueType.RowSizeOf([]uint64{uint64(a.AttentionValueLength), nKV, a.AttentionHeadCountKV})
				offloadAttnInc += rs
			} else {
				offloadAttnInc = uint64(0)
				for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.attn_(norm|q|qkv)\.weight`)) {
					var rs uint64
					switch {
					default: // norm.
						rs = GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nTokens})
						offloadAttnInc += rs
					case strings.HasSuffix(l.Name, ".attn_q.weight"):
						rs = GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[0], nTokens})
						offloadAttnInc += rs * 2 // Qcur, Qcur + RoPE.
						if !isOffloadOutputLayer {
							loadAttnInc = rs // Vcur.
						}
						rs = GGMLTypeF32.RowSizeOf([]uint64{nKV, nTokens, a.AttentionHeadCount})
						offloadAttnInc += rs // kq.
						rs = o.CacheKeyType.RowSizeOf([]uint64{uint64(a.AttentionKeyLength), nKV, a.AttentionHeadCountKV})
						offloadAttnInc += rs * 2 // k-?, v-?.
					case strings.HasSuffix(l.Name, ".attn_qkv.weight"):
						rs = GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[0], nTokens})
						offloadAttnInc += rs * 2 // Qcur, Qcur + RoPE.
						if !isOffloadOutputLayer {
							loadAttnInc = rs // Vcur.
						}
						rs = GGMLTypeF32.RowSizeOf([]uint64{nKV, nTokens, a.AttentionHeadCount})
						offloadAttnInc += rs // kq.
						rs = o.CacheKeyType.RowSizeOf([]uint64{uint64(a.AttentionKeyLength), nKV, a.AttentionHeadCountKV})
						offloadAttnInc += rs * 2 // k-?, v-?.
					}
				}
			}
			ffnInc := uint64(0)
			for _, l := range tfLs[len(tfLs)-1].Search(regexp.MustCompile(`.*\.\d+\.(attn_norm|ffn_norm|ffn_gate|ffn_up)\.weight`)) {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nTokens})
				ffnInc += rs
			}
			e.Load.Computation.Compute = GGUFBytesScalar(loadAttnInc)
			e.Offload.Computation.Compute = GGUFBytesScalar(max(offloadAttnInc, ffnInc))
			// Special case: we cannot use mmap for splitting expert weights in MoE.
			if a.ExpertCount > 0 {
				e.NoMMap = len(tfLs[0].Search(regexp.MustCompile(`.*\.\d+\.ffn_gate_exps\.weight`))) == 0
			}
		}
		// Finally, get the usage of output layer.
		{
			outInc := inpEmbd
			if a.Architecture == "mamba" {
				outInc += inpSMask + inpSSeq
			}
			if l, ok := opLs.Get("output.weight"); ok {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nTokens})
				outInc += rs
			} else if l, ok := ipLs.Get("token_embd.weight"); ok {
				rs := GGMLTypeF32.RowSizeOf([]uint64{l.Dimensions[l.NDimensions-1], nTokens})
				outInc += rs
			}
			outInc += uint64(e.Load.Weight.Output)
			e.Offload.Computation.Output = GGUFBytesScalar(outInc)
		}
	}

	return e
}

// Types for LLaMACpp estimated summary.
type (
	// LLaMACppUsageEstimateSummary represents the summary of the usage for loading the GGUF file in llama.cpp.
	LLaMACppUsageEstimateSummary struct {
		/* Basic */

		Memory []LLaMACppUsageEstimateMemorySummary `json:"memory"`

		/* Appendix */

		// Architecture describes what architecture this model implements.
		Architecture string `json:"architecture"`
		// ContextSize is the size of the context.
		ContextSize uint64 `json:"contextSize"`
		// FlashAttention is the flag to indicate whether enable the flash attention,
		// true for enable.
		FlashAttention bool `json:"flashAttention"`
		// NoMMap is the flag to indicate whether the file must be loaded without mmap,
		// true for total loaded.
		NoMMap bool `json:"noMMap"`
	}

	// LLaMACppUsageEstimateMemorySummary represents the memory summary of the usage for loading the GGUF file in llama.cpp.
	LLaMACppUsageEstimateMemorySummary struct {
		// OffloadLayers is the number of offloaded layers.
		OffloadLayers uint64 `json:"offloadLayers"`
		// FullOffloaded is the flag to indicate whether the layers are fully offloaded,
		// false for partial offloaded or zero offloaded.
		FullOffloaded bool `json:"fullOffloaded"`
		// UMA represents the usage of Unified Memory Architecture.
		UMA struct {
			// Load is the memory usage for loading the GGUF file in Load.
			RAM GGUFBytesScalar `json:"ram"`
			// VRAM is the memory usage for loading the GGUF file in VRAM.
			VRAM GGUFBytesScalar `json:"vram"`
		} `json:"uma"`
		// NonUMA represents the usage of Non-Unified Memory Architecture.
		NonUMA struct {
			// Load is the memory usage for loading the GGUF file in Load.
			RAM GGUFBytesScalar `json:"ram"`
			// VRAM is the memory usage for loading the GGUF file in VRAM.
			VRAM GGUFBytesScalar `json:"vram"`
		} `json:"nonUMA"`
	}
)

// SummarizeMemory returns the summary of the estimated memory usage of loading the GGUF file in llama.cpp,
// the input options are used to adjust the summary.
func (e LLaMACppUsageEstimate) SummarizeMemory(mmap bool, ramFootprint, vramFootprint uint64) (ems LLaMACppUsageEstimateMemorySummary) {
	ems.OffloadLayers, ems.FullOffloaded = e.OffloadLayers, e.FullOffloaded
	if ems.FullOffloaded {
		ems.OffloadLayers++ // The output layer is offloaded.
	}

	// UMA.
	{
		// RAM
		fp := e.Load.Footprint
		wg := e.Load.Weight.Sum()
		kv := e.Load.KVCache.Sum()
		cp := e.Load.Computation.Sum()
		ems.UMA.RAM = fp + wg + kv + cp
		if !e.NoMMap && mmap {
			ems.UMA.RAM -= wg
		}
		// VRAM.
		fp = e.Offload.Footprint
		wg = e.Offload.Weight.Sum()
		kv = e.Offload.KVCache.Sum()
		cp = 0
		ems.UMA.VRAM = fp + wg + kv + cp
	}

	// NonUMA.
	{
		// RAM.
		fp := GGUFBytesScalar(ramFootprint) + e.Load.Footprint
		wg := e.Load.Weight.Sum()
		kv := e.Load.KVCache.Sum()
		cp := e.Load.Computation.Sum()
		ems.NonUMA.RAM = fp + wg + kv + cp
		if !e.NoMMap && (mmap || e.FullOffloaded) {
			ems.NonUMA.RAM -= wg
			if !mmap {
				ems.NonUMA.RAM += e.Load.Weight.Output
			}
		}
		// VRAM.
		fp = GGUFBytesScalar(vramFootprint) + e.Offload.Footprint
		wg = e.Offload.Weight.Sum()
		kv = e.Offload.KVCache.Sum()
		cp = e.Offload.Computation.Sum()
		ems.NonUMA.VRAM = fp + wg + kv + cp
	}

	return ems
}

// Summarize returns the summary of the estimated result of loading the GGUF file in llama.cpp,
// the input options are used to adjust the summary.
func (e LLaMACppUsageEstimate) Summarize(mmap bool, ramFootprint, vramFootprint uint64) (es LLaMACppUsageEstimateSummary) {
	// Summarize memory.
	es.Memory = []LLaMACppUsageEstimateMemorySummary{
		e.SummarizeMemory(mmap, ramFootprint, vramFootprint),
	}

	// Just copy from the original estimate.
	es.Architecture = e.Architecture
	es.ContextSize = e.ContextSize
	es.FlashAttention = e.FlashAttention
	es.NoMMap = e.NoMMap

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
