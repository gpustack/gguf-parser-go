package gguf_parser

import (
	"regexp"
	"strings"

	"github.com/gpustack/gguf-parser-go/util/ptr"
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
		// NoMMap is the flag to indicate whether support the mmap,
		// true for support.
		NoMMap bool `json:"noMMap"`
		// EmbeddingOnly is the flag to indicate whether the model is used for embedding only,
		// true for embedding only.
		EmbeddingOnly bool `json:"embeddingOnly"`
		// LogicalBatchSize is the logical batch size.
		LogicalBatchSize int32 `json:"logicalBatchSize"`
		// PhysicalBatchSize is the physical batch size.
		PhysicalBatchSize int32 `json:"physicalBatchSize"`
		// Devices represents the memory usage for running the GGUF file,
		// the first device is the CPU, and the rest are GPUs.
		Devices []LLaMACppMemoryUsage `json:"devices"`
		// MultimodalProjector is the memory usage of multimodal projector.
		MultimodalProjector *LLaMACppUsageEstimate `json:"multimodalProjector,omitempty"`
		// Drafter is the memory usage of drafter.
		Drafter *LLaMACppUsageEstimate `json:"drafter,omitempty"`
	}

	// LLaMACppMemoryUsage represents the memory usage for expanding the GGUF file in llama.cpp.
	LLaMACppMemoryUsage struct {
		// HandleLayers is the number of layers that the device can handle.
		HandleLayers uint64 `json:"handleLayers"`
		// LastLayer is the index of the last layer the device can handle.
		LastLayer int `json:"latestLayer"`
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
	if o.LogicalBatchSize == nil {
		o.LogicalBatchSize = ptr.To(int32(2048))
	} else {
		// See https://github.com/ggerganov/llama.cpp/blob/0bf16de07b0692e7df26b9a633e232bbd66e0360/src/llama.cpp#L16519-L16525.
		o.LogicalBatchSize = ptr.To(max(32, *o.LogicalBatchSize))
	}
	if o.PhysicalBatchSize == nil {
		o.PhysicalBatchSize = ptr.To(int32(512))
	}
	if *o.PhysicalBatchSize > *o.LogicalBatchSize {
		panic("physical batch size must be less than or equal to logical batch size")
	}
	if o.SplitMode >= _LLAMACppSplitModeMax {
		panic("split mode must be less than max")
	}
	switch {
	case o.TensorSplitFraction == nil:
		o.TensorSplitFraction = []float64{1}
		o.MainGPUIndex = 0
	case o.MainGPUIndex < 0 || o.MainGPUIndex >= len(o.TensorSplitFraction):
		panic("main device index must be range of 0 to the length of tensor split fraction")
	}

	// Devices.
	e.Devices = make([]LLaMACppMemoryUsage, len(o.TensorSplitFraction)+1)
	for i := range e.Devices {
		e.Devices[i].LastLayer = -1
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

	// Embedding.
	if !a.AttentionCausal {
		e.EmbeddingOnly = true
		o.PhysicalBatchSize = o.LogicalBatchSize
	}

	// Batch size.
	e.LogicalBatchSize = *o.LogicalBatchSize
	e.PhysicalBatchSize = *o.PhysicalBatchSize

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
		if o.InMaxContextSize {
			nContext = min(nContext, a.MaximumContextLength)
		}
		// Padding context size,
		// see https://github.com/ggerganov/llama.cpp/blob/278d0e18469aacf505be18ce790a63c7cc31be26/src/llama.cpp#L19001-L19002.
		if o.FlashAttention {
			nContext = GGMLPadding(nContext, 256)
		} else {
			nContext = GGMLPadding(nContext, 32)
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

		fullOffload, partialOffload, zeroOffload bool
	)
	{
		// For clip,
		// see https://github.com/ggerganov/llama.cpp/blob/148ec970b62c3c5ae0a8bfdaad2fc237aaae350d/examples/llava/clip.cpp#L994-L1008.
		if a.Architecture == "clip" {
			o.OffloadLayers = ptr.To(a.BlockCount + 1) // Clip means full offload.
		}
		switch v := o.OffloadLayers; {
		case v == nil:
			o.OffloadLayers = ptr.To(a.BlockCount)
			nOffloadLayers = a.BlockCount
			isOffloadOutputLayer = true
		case *v != 0:
			nOffloadLayers = *v
			if nOffloadLayers > a.BlockCount {
				isOffloadOutputLayer = true
				nOffloadLayers = a.BlockCount
			}
		}
		nLoadLayers -= nOffloadLayers

		e.FullOffloaded = isOffloadOutputLayer && nLoadLayers == 0
		e.OffloadLayers = nOffloadLayers

		fullOffload = isOffloadOutputLayer && nLoadLayers == 0
		partialOffload = nLoadLayers > 0 && nOffloadLayers > 0
		zeroOffload = !fullOffload && !partialOffload
	}

	// Footprint.
	{
		// Bootstrap.
		e.Devices[0].Footprint = GGUFBytesScalar(5*1024*1024) /* model load */ + (gf.Size - gf.ModelSize) /* metadata */

		// Tokens,
		// https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L6380-L6384.
		fp := t.TokensLength * (4 /* token type */ + 4 /* token score*/)
		if t.Model == "gpt2" {
			fp += t.MergesLength * (48 /* key type */ + 56 /* value type */)
		}
		fp += t.TokensLength * (32 /* id to token vector */ + (24 + 32) /* token to id map*/)
		e.Devices[0].Footprint += GGUFBytesScalar(fp)

		// Output buffer,
		// see https://github.com/ggerganov/llama.cpp/blob/7672adeec7a79ea271058c63106c142ba84f951a/llama.cpp#L11940-L12003.
		ob := 4 /* float32 size */ * (a.VocabularyLength + a.EmbeddingLength) * nParallel
		if fullOffload {
			e.Devices[o.MainGPUIndex+1].Footprint += GGUFBytesScalar(ob)
		} else {
			e.Devices[0].Footprint += GGUFBytesScalar(ob)
		}
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
		switch a.Architecture {
		case "clip":
			e.Devices[1].Weight.Compute = GGUFBytesScalar(ls.Bytes())
		default:
			for i, j, offloadStart := 0, 0, len(tfLs)-int(nOffloadLayers); i < len(tfLs); i++ {
				switch {
				case i < int(nLoadLayers):
					e.Devices[0].HandleLayers += 1
					e.Devices[0].LastLayer = i
					e.Devices[0].Weight.Compute += GGUFBytesScalar(tfLs[i].Bytes())
				case i >= offloadStart:
					x := float64(i-offloadStart) / float64(nOffloadLayers)
					for k := j; k < len(o.TensorSplitFraction); k++ {
						if x < o.TensorSplitFraction[k] {
							j = k
							break
						}
					}
					e.Devices[j+1].HandleLayers += 1
					e.Devices[j+1].LastLayer = i
					e.Devices[j+1].Weight.Compute += GGUFBytesScalar(tfLs[i].Bytes())
				}
			}
		}

		// IO,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L4930-L5002.
		e.Devices[0].Weight.Input = GGUFBytesScalar(ipLs.Bytes())
		if _, ok := opLs.Get("output.weight"); ok {
			e.Devices[0].Weight.Output = GGUFBytesScalar(opLs.Bytes())
		} else if a.AttentionCausal {
			e.Devices[0].Weight.Output = GGUFBytesScalar(opLs.Bytes()) + e.Devices[0].Weight.Input /* duplicate the input layer */
		}
	}

	// KV cache,
	// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L2479-L2501.
	{
		krs := o.CacheKeyType.RowSizeOf([]uint64{a.EmbeddingKeyGQA * nKV})
		vrs := o.CacheValueType.RowSizeOf([]uint64{a.EmbeddingValueGQA * nKV})

		e.Devices[0].KVCache.Key = GGUFBytesScalar(krs * nLoadLayers)
		e.Devices[0].KVCache.Value = GGUFBytesScalar(vrs * nLoadLayers)
		if !*o.OffloadKVCache {
			e.Devices[0].KVCache.Key += GGUFBytesScalar(krs * nOffloadLayers)
			e.Devices[0].KVCache.Value += GGUFBytesScalar(vrs * nOffloadLayers)
		} else if !zeroOffload {
			for i, d := range e.Devices[1:] {
				e.Devices[i+1].KVCache.Key = GGUFBytesScalar(krs * d.HandleLayers)
				e.Devices[i+1].KVCache.Value = GGUFBytesScalar(vrs * d.HandleLayers)
			}
		}
	}

	// Computation.
	{
		// Bootstrap, compute metadata,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L16135-L16136.
		cm := GGMLTensorOverhead()*GGMLComputationGraphNodesMaximum +
			GGMLComputationGraphOverhead(GGMLComputationGraphNodesMaximum, false)
		e.Devices[0].Computation.Footprint = GGUFBytesScalar(cm)

		// Scheduler overhead,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L16149.
		e.Devices[0].Computation.Footprint += GGUFBytesScalar(4 * 1024 * 1024)

		// GGML context,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L5015-L5036.
		gc := 2 /* buffer count */ * GGMLTensorOverhead() * (uint64(len(gf.TensorInfos)) + 1 + a.BlockCount*3)
		e.Devices[0].Computation.Footprint += GGUFBytesScalar(gc)

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
		switch a.Architecture {
		case "clip":
			// NOP.
		case "mamba":
			e.Devices[0].Computation.Input = GGUFBytesScalar(inpTokens + inpEmbd + inpSMask + inpSSeq + inpOutIds)
			if !zeroOffload {
				v := GGUFBytesScalar(inpEmbd + inpSMask + inpSSeq + inpOutIds)
				for i := range e.Devices[1:] {
					e.Devices[i+1].Computation.Input += v
				}
			}
		default:
			e.Devices[0].Computation.Input = GGUFBytesScalar(inpTokens + inpEmbd + inpPos + inpKQMask + inpOutIds)
			if !zeroOffload {
				v := GGUFBytesScalar(inpEmbd + inpPos + inpKQMask + inpOutIds)
				for i := range e.Devices[1:] {
					e.Devices[i+1].Computation.Input += v
				}
			}
		}
		// Since the steps between transformer layers are serial,
		// the allocated memory can be reused for the next layer.
		// So, we only consider the usage of the largest layer,
		// which is the last layer by default.
		switch a.Architecture {
		case "clip":
			// NOP.
		case "mamba":
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
			for i, d := range e.Devices[1:] {
				if d.LastLayer < 0 {
					continue
				}
				e.Devices[i+1].Computation.Compute = GGUFBytesScalar(convInc + ssmInc)
			}
		default:
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
			e.Devices[0].Computation.Compute = GGUFBytesScalar(loadAttnInc)
			for i, d := range e.Devices[1:] {
				if d.LastLayer < 0 {
					continue
				}
				e.Devices[i+1].Computation.Compute = GGUFBytesScalar(max(offloadAttnInc, ffnInc))
			}
			// Special case: we cannot use mmap for splitting expert weights in MoE.
			if a.ExpertCount > 0 {
				e.NoMMap = len(tfLs[0].Search(regexp.MustCompile(`.*\.\d+\.ffn_gate_exps\.weight`))) == 0
			}
		}
		// Finally, get the usage of output layer.
		switch a.Architecture {
		case "clip":
			// NOP.
		default:
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
			outInc += uint64(e.Devices[0].Weight.Output)
			e.Devices[o.MainGPUIndex+1].Computation.Output += GGUFBytesScalar(outInc)
		}
	}

	// Multimodal projector.
	e.MultimodalProjector = o.MultimodalProjector

	// Drafter.
	e.Drafter = o.Drafter

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
		// EmbeddingOnly is the flag to indicate whether the model is used for embedding only,
		// true for embedding only.
		EmbeddingOnly bool `json:"embeddingOnly"`
		// LogicalBatchSize is the logical batch size.
		LogicalBatchSize int32 `json:"logicalBatchSize"`
		// PhysicalBatchSize is the physical batch size.
		PhysicalBatchSize int32 `json:"physicalBatchSize"`
	}

	// LLaMACppUsageEstimateMemorySummary represents the memory summary of the usage for loading the GGUF file in llama.cpp.
	LLaMACppUsageEstimateMemorySummary struct {
		// OffloadLayers is the number of offloaded layers.
		OffloadLayers uint64 `json:"offloadLayers"`
		// FullOffloaded is the flag to indicate whether the layers are fully offloaded,
		// false for partial offloaded or zero offloaded.
		FullOffloaded bool `json:"fullOffloaded"`
		// RAM is the memory usage for loading the GGUF file in RAM.
		RAM LLaMACppUsageEstimateMemoryDetail `json:"ram"`
		// VRAMs is the memory usage for loading the GGUF file in VRAM per device.
		VRAMs []LLaMACppUsageEstimateMemoryDetail `json:"vrams"`
	}

	// LLaMACppUsageEstimateMemoryDetail represents the detailed memory usage for loading the GGUF file in llama.cpp.
	LLaMACppUsageEstimateMemoryDetail struct {
		// UMA represents the usage of Unified Memory Architecture.
		UMA GGUFBytesScalar `json:"uma"`
		// NonUMA represents the usage of Non-Unified Memory Architecture.
		NonUMA GGUFBytesScalar `json:"nonuma"`
	}
)

// SummarizeMemory returns the summary of the estimated memory usage of loading the GGUF file in llama.cpp,
// the input options are used to adjust the summary.
func (e LLaMACppUsageEstimate) SummarizeMemory(mmap bool, nonUMARamFootprint, nonUMAVramFootprint uint64) (ems LLaMACppUsageEstimateMemorySummary) {
	ems.OffloadLayers, ems.FullOffloaded = e.OffloadLayers, e.FullOffloaded
	if ems.FullOffloaded {
		ems.OffloadLayers++ // The output layer is offloaded.
	}

	ems.VRAMs = make([]LLaMACppUsageEstimateMemoryDetail, len(e.Devices)-1)

	// RAM.
	{
		fp := e.Devices[0].Footprint
		wg := e.Devices[0].Weight.Sum()
		kv := e.Devices[0].KVCache.Sum()
		cp := e.Devices[0].Computation.Sum()

		// UMA.
		ems.RAM.UMA = fp + wg + kv + cp
		if !e.NoMMap && (mmap || e.FullOffloaded) {
			ems.RAM.UMA -= wg
			if !mmap {
				ems.RAM.UMA += e.Devices[0].Weight.Output
			}
		}

		// NonUMA.
		ems.RAM.NonUMA = GGUFBytesScalar(nonUMARamFootprint) + ems.RAM.UMA
	}

	// VRAMs.
	{
		for i, v := range e.Devices[1:] {
			fp := v.Footprint
			wg := v.Weight.Sum()
			kv := v.KVCache.Sum()
			cp := v.Computation.Sum()

			// UMA.
			ems.VRAMs[i].UMA = fp + wg + kv + /* cp */ 0
			if !e.NoMMap && mmap {
				ems.VRAMs[i].UMA -= wg
			}

			// NonUMA.
			ems.VRAMs[i].NonUMA = GGUFBytesScalar(nonUMAVramFootprint) + fp + wg + kv + cp
		}
	}

	// MultimodalProjector.
	if e.MultimodalProjector != nil {
		cems := e.MultimodalProjector.SummarizeMemory(mmap, 0, 0)
		ems.RAM.UMA += cems.RAM.UMA
		ems.RAM.NonUMA += cems.RAM.NonUMA
		for i, v := range cems.VRAMs {
			ems.VRAMs[i].UMA += v.UMA
			ems.VRAMs[i].NonUMA += v.NonUMA
		}
	}

	// Drafter.
	if e.Drafter != nil {
		dmes := e.Drafter.SummarizeMemory(mmap, 0, 0)
		ems.RAM.UMA += dmes.RAM.UMA
		ems.RAM.NonUMA += dmes.RAM.NonUMA
		for i, v := range dmes.VRAMs {
			ems.VRAMs[i].UMA += v.UMA
			ems.VRAMs[i].NonUMA += v.NonUMA
		}
	}

	return ems
}

// Summarize returns the summary of the estimated result of loading the GGUF file in llama.cpp,
// the input options are used to adjust the summary.
func (e LLaMACppUsageEstimate) Summarize(mmap bool, nonUMARamFootprint, nonUMAVramFootprint uint64) (es LLaMACppUsageEstimateSummary) {
	// Summarize memory.
	es.Memory = []LLaMACppUsageEstimateMemorySummary{
		e.SummarizeMemory(mmap, nonUMARamFootprint, nonUMAVramFootprint),
	}

	// Just copy from the original estimate.
	es.Architecture = e.Architecture
	es.ContextSize = e.ContextSize
	es.FlashAttention = e.FlashAttention
	es.NoMMap = e.NoMMap
	es.EmbeddingOnly = e.EmbeddingOnly
	es.LogicalBatchSize = e.LogicalBatchSize
	es.PhysicalBatchSize = e.PhysicalBatchSize

	return es
}

func (u LLaMACppWeightUsage) Sum() GGUFBytesScalar {
	return u.Input + u.Compute + u.Output
}

func (u LLaMACppKVCacheUsage) Sum() GGUFBytesScalar {
	return u.Key + u.Value
}

func (u LLaMACppComputationUsage) Sum() GGUFBytesScalar {
	return u.Footprint + max(u.Input+u.Compute, u.Output)
}
