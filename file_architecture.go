package gguf_parser

// GGUFArchitectureMetadata represents the architecture metadata of a GGUF file.
type GGUFArchitectureMetadata struct {
	/* Basic */

	// Architecture describes what architecture this model implements.
	//
	// All lowercase ASCII, with only [a-z0-9]+ characters allowed.
	Architecture string `json:"architecture"`
	// MaximumContextLength(n_ctx_train) is the maximum context length of the model.
	//
	// For most architectures, this is the hard limit on the length of the input.
	// Architectures, like RWKV,
	// that are not reliant on transformer-style attention may be able to handle larger inputs,
	// but this is not guaranteed.
	MaximumContextLength uint64 `json:"maximumContextLength"`
	// EmbeddingLength(n_embd) is the length of the embedding layer.
	EmbeddingLength uint64 `json:"embeddingLength"`
	// BlockCount(n_layer) is the number of blocks of attention and feed-forward layers,
	// i.e. the bulk of the LLM.
	// This does not include the input or embedding layers.
	BlockCount uint64 `json:"blockCount"`
	// FeedForwardLength(n_ff) is the length of the feed-forward layer.
	FeedForwardLength uint64 `json:"feedForwardLength,omitempty"`
	// ExpertCount(n_expert) is the number of experts in MoE models.
	ExpertCount uint32 `json:"expertCount,omitempty"`
	// ExpertUsedCount(n_expert_used) is the number of experts used during each token evaluation in MoE models.
	ExpertUsedCount uint32 `json:"expertUsedCount,omitempty"`
	// AttentionHeadCount(n_head) is the number of attention heads.
	AttentionHeadCount uint64 `json:"attentionHeadCount,omitempty"`
	// AttentionHeadCountKV(n_head_kv) is the number of attention heads per group used in Grouped-Query-Attention.
	//
	// If not provided or equal to AttentionHeadCount,
	// the model does not use Grouped-Query-Attention.
	AttentionHeadCountKV uint64 `json:"attentionHeadCountKV,omitempty"`
	// AttentionMaxALiBIBias is the maximum bias to use for ALiBI.
	AttentionMaxALiBIBias float32 `json:"attentionMaxALiBIBias,omitempty"`
	// AttentionClampKQV describes a value `C`,
	// which is used to clamp the values of the `Q`, `K` and `V` tensors between `[-C, C]`.
	AttentionClampKQV float32 `json:"attentionClampKQV,omitempty"`
	// AttentionLayerNormEpsilon is the epsilon value used in the LayerNorm(Layer Normalization).
	AttentionLayerNormEpsilon float32 `json:"attentionLayerNormEpsilon,omitempty"`
	// AttentionLayerNormRMSEpsilon is the epsilon value used in the RMSNorm(Root Mean Square Layer Normalization),
	// which is a simplification of the original LayerNorm.
	AttentionLayerNormRMSEpsilon float32 `json:"attentionLayerNormRMSEpsilon,omitempty"`
	// AttentionKeyLength is the size of a key head.
	//
	// Defaults to `EmbeddingLength / AttentionHeadCount`.
	AttentionKeyLength uint32 `json:"attentionKeyLength"`
	// AttentionValueLength is the size of a value head.
	//
	// Defaults to `EmbeddingLength / AttentionHeadCount`.
	AttentionValueLength uint32 `json:"attentionValueLength"`
	// RoPEDimensionCount is the number of dimensions in the RoPE(Rotary Positional Encoding).
	RoPEDimensionCount uint64 `json:"ropeDimensionCount,omitempty"`
	// RoPEFrequencyBase is the base frequency of the RoPE.
	RoPEFrequencyBase float32 `json:"ropeFrequencyBase,omitempty"`
	// RoPEFrequencyScale is the frequency scale of the RoPE.
	RoPEScalingType string `json:"ropeScalingType,omitempty"`
	// RoPEScalingFactor is the scaling factor of the RoPE.
	RoPEScalingFactor float32 `json:"ropeScalingFactor,omitempty"`
	// RoPEScalingOriginalContextLength is the original context length of the RoPE scaling.
	RoPEScalingOriginalContextLength uint64 `json:"ropeScalingOriginalContextLength,omitempty"`
	// RoPEScalingFinetuned is true if the RoPE scaling is fine-tuned.
	RoPEScalingFinetuned bool `json:"ropeScalingFinetuned,omitempty"`
	// SSMConvolutionKernel is the size of the convolution kernel used in the SSM(Selective State Space Model).
	SSMConvolutionKernel uint32 `json:"ssmConvolutionKernel,omitempty"`
	// SSMInnerSize is the embedding size of the state in SSM.
	SSMInnerSize uint32 `json:"ssmInnerSize,omitempty"`
	// SSMStateSize is the size of the recurrent state in SSM.
	SSMStateSize uint32 `json:"ssmStateSize,omitempty"`
	// SSMTimeStepRank is the rank of the time steps in SSM.
	SSMTimeStepRank uint32 `json:"ssmTimeStepRank,omitempty"`
	// VocabularyLength is the size of the vocabulary.
	//
	// VocabularyLength is the same as the tokenizer's token size.
	VocabularyLength uint64 `json:"vocabularyLength"`

	/* Appendix */

	// EmbeddingKeyGQA is the number of key GQA in the embedding layer.
	EmbeddingKeyGQA uint64 `json:"embeddingKeyGQA,omitempty"`
	// EmbeddingValueGQA is the number of value GQA in the embedding layer.
	EmbeddingValueGQA uint64 `json:"embeddingValueGQA,omitempty"`
	// EmbeddingGGQA is the GQA of the embedding layer.
	EmbeddingGQA uint64 `json:"embeddingGQA,omitempty"`
}

// Architecture returns the architecture metadata of the GGUF file.
func (gf *GGUFFile) Architecture() (ga GGUFArchitectureMetadata) {
	arch := "llama"
	if v, ok := gf.Header.MetadataKV.Get("general.architecture"); ok {
		arch = v.ValueString()
	}
	var (
		contextLengthKey     = arch + ".context_length"
		embeddingLengthKey   = arch + ".embedding_length"
		blockCountKey        = arch + ".block_count"
		feedForwardLengthKey = arch + ".feed_forward_length"
		expertCountKey       = arch + ".expert_count"
		expertUsedCountKey   = arch + ".expert_used_count"

		attentionHeadCountKey           = arch + ".attention.head_count"
		attentionHeadCountKVKey         = arch + ".attention.head_count_kv"
		attentionMaxALiBIBiasKey        = arch + ".attention.max_alibi_bias"
		attentionMaxALiBIBiasKey2       = arch + ".attention.alibi_bias_max"
		attentionClampKQVKey            = arch + ".attention.clamp_kqv"
		attentionClampKQVKey2           = arch + ".attention.clip_kqv"
		attentionLayerNormEpsilonKey    = arch + ".attention.layer_norm_epsilon"
		attentionLayerNormRMSEpsilonKey = arch + ".attention.layer_norm_rms_epsilon"
		attentionKeyLengthKey           = arch + ".attention.key_length"
		attentionValueLengthKey         = arch + ".attention.value_length"

		ropeDimensionCountKey         = arch + ".rope.dimension_count"
		ropeFrequencyBaseKey          = arch + ".rope.freq_base"
		ropeScaleLinearKey            = arch + ".rope.scale_linear"
		ropeScalingTypeKey            = arch + ".rope.scaling.type"
		ropeScalingFactorKey          = arch + ".rope.scaling.factor"
		ropeScalingOriginalContextKey = arch + ".rope.scaling.original_context_length" // uint32 maybe
		ropeScalingFinetunedKey       = arch + ".rope.scaling.finetuned"

		ssmConvolutionKernelKey = arch + ".ssm.conv_kernel"
		ssmInnerSizeKey         = arch + ".ssm.inner_size"
		ssmStateSizeKey         = arch + ".ssm.state_size"
		ssmTimeStepRankKey      = arch + ".ssm.time_step_rank"

		vocabularyLengthKey    = arch + ".vocab_size"
		tokenizerGGMLTokensKey = "tokenizer.ggml.tokens"
	)

	ga.Architecture = arch

	m, _ := gf.Header.MetadataKV.Index([]string{
		contextLengthKey,
		embeddingLengthKey,
		blockCountKey,
		feedForwardLengthKey,
		expertCountKey,
		expertUsedCountKey,
		attentionHeadCountKey,
		attentionHeadCountKVKey,
		attentionMaxALiBIBiasKey,
		attentionMaxALiBIBiasKey2,
		attentionClampKQVKey,
		attentionClampKQVKey2,
		attentionLayerNormEpsilonKey,
		attentionLayerNormRMSEpsilonKey,
		attentionKeyLengthKey,
		attentionValueLengthKey,
		ropeDimensionCountKey,
		ropeFrequencyBaseKey,
		ropeScaleLinearKey,
		ropeScalingTypeKey,
		ropeScalingFactorKey,
		ropeScalingOriginalContextKey,
		ropeScalingFinetunedKey,
		ssmConvolutionKernelKey,
		ssmInnerSizeKey,
		ssmStateSizeKey,
		ssmTimeStepRankKey,
		vocabularyLengthKey,
		tokenizerGGMLTokensKey,
	})

	if v, ok := m[contextLengthKey]; ok {
		ga.MaximumContextLength = ValueNumeric[uint64](v)
	}
	if v, ok := m[embeddingLengthKey]; ok {
		ga.EmbeddingLength = ValueNumeric[uint64](v)
	}
	if v, ok := m[blockCountKey]; ok {
		ga.BlockCount = ValueNumeric[uint64](v)
	}
	if v, ok := m[feedForwardLengthKey]; ok {
		ga.FeedForwardLength = ValueNumeric[uint64](v)
	}
	if v, ok := m[expertCountKey]; ok {
		ga.ExpertCount = ValueNumeric[uint32](v)
	}
	if v, ok := m[expertUsedCountKey]; ok {
		ga.ExpertUsedCount = ValueNumeric[uint32](v)
	}

	if v, ok := m[attentionHeadCountKey]; ok {
		ga.AttentionHeadCount = ValueNumeric[uint64](v)
	}
	if v, ok := m[attentionHeadCountKVKey]; ok {
		ga.AttentionHeadCountKV = ValueNumeric[uint64](v)
	} else {
		ga.AttentionHeadCountKV = ga.AttentionHeadCount
	}
	if v, ok := m[attentionMaxALiBIBiasKey]; ok {
		ga.AttentionMaxALiBIBias = ValueNumeric[float32](v)
	} else if v, ok := m[attentionMaxALiBIBiasKey2]; ok {
		ga.AttentionMaxALiBIBias = ValueNumeric[float32](v)
	}
	if v, ok := m[attentionClampKQVKey]; ok {
		ga.AttentionClampKQV = ValueNumeric[float32](v)
	} else if v, ok := m[attentionClampKQVKey2]; ok {
		ga.AttentionClampKQV = ValueNumeric[float32](v)
	}
	if v, ok := m[attentionLayerNormEpsilonKey]; ok {
		ga.AttentionLayerNormEpsilon = ValueNumeric[float32](v)
	}
	if v, ok := m[attentionLayerNormRMSEpsilonKey]; ok {
		ga.AttentionLayerNormRMSEpsilon = ValueNumeric[float32](v)
	}
	if v, ok := m[attentionKeyLengthKey]; ok {
		ga.AttentionKeyLength = ValueNumeric[uint32](v)
	} else if ga.AttentionHeadCount != 0 {
		ga.AttentionKeyLength = uint32(ga.EmbeddingLength / ga.AttentionHeadCount)
	}
	if v, ok := m[attentionValueLengthKey]; ok {
		ga.AttentionValueLength = ValueNumeric[uint32](v)
	} else if ga.AttentionHeadCount != 0 {
		ga.AttentionValueLength = uint32(ga.EmbeddingLength / ga.AttentionHeadCount)
	}

	if v, ok := m[ropeDimensionCountKey]; ok {
		ga.RoPEDimensionCount = ValueNumeric[uint64](v)
	}
	if v, ok := m[ropeFrequencyBaseKey]; ok {
		ga.RoPEFrequencyBase = ValueNumeric[float32](v)
	}
	if v, ok := m[ropeScaleLinearKey]; ok {
		ga.RoPEScalingType = "linear"
		ga.RoPEScalingFactor = ValueNumeric[float32](v)
	}
	if v, ok := m[ropeScalingTypeKey]; ok {
		ga.RoPEScalingType = v.ValueString()
	}
	if v, ok := m[ropeScalingFactorKey]; ok {
		ga.RoPEScalingFactor = ValueNumeric[float32](v)
	}
	if v, ok := m[ropeScalingOriginalContextKey]; ok {
		ga.RoPEScalingOriginalContextLength = ValueNumeric[uint64](v)
	}
	if v, ok := m[ropeScalingFinetunedKey]; ok {
		ga.RoPEScalingFinetuned = v.ValueBool()
	}

	if v, ok := m[ssmConvolutionKernelKey]; ok {
		ga.SSMConvolutionKernel = ValueNumeric[uint32](v)
	}
	if v, ok := m[ssmInnerSizeKey]; ok {
		ga.SSMInnerSize = ValueNumeric[uint32](v)
	}
	if v, ok := m[ssmStateSizeKey]; ok {
		ga.SSMStateSize = ValueNumeric[uint32](v)
	}
	if v, ok := m[ssmTimeStepRankKey]; ok {
		ga.SSMTimeStepRank = ValueNumeric[uint32](v)
	}

	if v, ok := m[vocabularyLengthKey]; ok {
		ga.VocabularyLength = ValueNumeric[uint64](v)
	} else if v, ok := m[tokenizerGGMLTokensKey]; ok {
		ga.VocabularyLength = v.ValueArray().Len
	}

	{
		if ga.AttentionHeadCount > 0 {
			ga.EmbeddingKeyGQA = uint64(ga.AttentionKeyLength) * ga.AttentionHeadCountKV
			ga.EmbeddingValueGQA = uint64(ga.AttentionValueLength) * ga.AttentionHeadCountKV
		}
		if ga.Architecture == "mamba" {
			ga.EmbeddingKeyGQA = uint64((ga.SSMConvolutionKernel - 1) * ga.SSMInnerSize)
			ga.EmbeddingValueGQA = uint64(ga.SSMStateSize * ga.SSMInnerSize)
		}
		ga.EmbeddingGQA = ga.EmbeddingValueGQA
	}

	return ga
}
