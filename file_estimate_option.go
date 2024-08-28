package gguf_parser

import (
	"slices"

	"github.com/gpustack/gguf-parser-go/util/ptr"
)

type (
	_LLaMACppRunEstimateOptions struct {
		Architecture        *GGUFArchitecture
		Tokenizer           *GGUFTokenizer
		ContextSize         *int32
		InMaxContextSize    bool
		LogicalBatchSize    *int32
		PhysicalBatchSize   *int32
		ParallelSize        *int32
		CacheKeyType        *GGMLType
		CacheValueType      *GGMLType
		OffloadKVCache      *bool
		OffloadLayers       *uint64
		FlashAttention      bool
		SplitMode           LLaMACppSplitMode
		TensorSplitFraction []float64
		MainGPUIndex        int
		RPCServers          []string
		Projector           *LLaMACppRunEstimate
		Drafter             *LLaMACppRunEstimate
		Adapters            []LLaMACppRunEstimate
	}
	LLaMACppRunEstimateOption func(*_LLaMACppRunEstimateOptions)
)

// WithArchitecture sets the architecture for the estimate.
//
// Allows reusing the same GGUFArchitecture for multiple estimates.
func WithArchitecture(arch GGUFArchitecture) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.Architecture = &arch
	}
}

// WithTokenizer sets the tokenizer for the estimate.
//
// Allows reusing the same GGUFTokenizer for multiple estimates.
func WithTokenizer(tokenizer GGUFTokenizer) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.Tokenizer = &tokenizer
	}
}

// WithContextSize sets the context size for the estimate.
func WithContextSize(size int32) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if size <= 0 {
			return
		}
		o.ContextSize = &size
	}
}

// WithinMaxContextSize limits the context size to the maximum,
// if the context size is over the maximum.
func WithinMaxContextSize() LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.InMaxContextSize = true
	}
}

// WithLogicalBatchSize sets the logical batch size for the estimate.
func WithLogicalBatchSize(size int32) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if size <= 0 {
			return
		}
		o.LogicalBatchSize = &size
	}
}

// WithPhysicalBatchSize sets the physical batch size for the estimate.
func WithPhysicalBatchSize(size int32) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if size <= 0 {
			return
		}
		o.PhysicalBatchSize = &size
	}
}

// WithParallelSize sets the (decoding sequences) parallel size for the estimate.
func WithParallelSize(size int32) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if size <= 0 {
			return
		}
		o.ParallelSize = &size
	}
}

// _GGUFEstimateCacheTypeAllowList is the allow list of cache key and value types.
var _GGUFEstimateCacheTypeAllowList = []GGMLType{
	GGMLTypeF32,
	GGMLTypeF16,
	GGMLTypeQ8_0,
	GGMLTypeQ4_0, GGMLTypeQ4_1,
	GGMLTypeIQ4_NL,
	GGMLTypeQ5_0, GGMLTypeQ5_1,
}

// WithCacheKeyType sets the cache key type for the estimate.
func WithCacheKeyType(t GGMLType) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if slices.Contains(_GGUFEstimateCacheTypeAllowList, t) {
			o.CacheKeyType = &t
		}
	}
}

// WithCacheValueType sets the cache value type for the estimate.
func WithCacheValueType(t GGMLType) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if slices.Contains(_GGUFEstimateCacheTypeAllowList, t) {
			o.CacheValueType = &t
		}
	}
}

// WithoutOffloadKVCache disables offloading the KV cache.
func WithoutOffloadKVCache() LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.OffloadKVCache = ptr.To(false)
	}
}

// WithOffloadLayers sets the number of layers to offload.
func WithOffloadLayers(layers uint64) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.OffloadLayers = &layers
	}
}

// WithFlashAttention sets the flash attention flag.
func WithFlashAttention() LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.FlashAttention = true
	}
}

// LLaMACppSplitMode is the split mode for LLaMACpp.
type LLaMACppSplitMode uint

const (
	LLaMACppSplitModeLayer LLaMACppSplitMode = iota
	LLaMACppSplitModeRow
	LLaMACppSplitModeNone
	_LLAMACppSplitModeMax
)

// WithSplitMode sets the split mode for the estimate.
func WithSplitMode(mode LLaMACppSplitMode) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if mode < _LLAMACppSplitModeMax {
			o.SplitMode = mode
		}
	}
}

// WithTensorSplitFraction sets the tensor split cumulative fractions for the estimate.
//
// WithTensorSplitFraction accepts a variadic number of fractions,
// all fraction values must be in the range of [0, 1],
// and the last fraction must be 1.
//
// For example, WithTensorSplitFraction(0.2, 0.4, 0.6, 0.8, 1) will split the tensor into five parts with 20% each.
func WithTensorSplitFraction(fractions []float64) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if len(fractions) == 0 {
			return
		}
		for _, f := range fractions {
			if f < 0 || f > 1 {
				return
			}
		}
		if fractions[len(fractions)-1] != 1 {
			return
		}
		o.TensorSplitFraction = fractions
	}
}

// WithMainGPUIndex sets the main device for the estimate.
//
// When split mode is LLaMACppSplitModeNone, the main device is the only device.
// When split mode is LLaMACppSplitModeRow, the main device handles the intermediate results and KV.
//
// WithMainGPUIndex only works when TensorSplitFraction is set.
func WithMainGPUIndex(di int) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.MainGPUIndex = di
	}
}

// WithRPCServers sets the RPC servers for the estimate.
func WithRPCServers(srvs []string) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if len(srvs) == 0 {
			return
		}
		o.RPCServers = srvs
	}
}

// WithDrafter sets the drafter estimate usage.
func WithDrafter(dft *LLaMACppRunEstimate) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.Drafter = dft
	}
}

// WithProjector sets the multimodal projector estimate usage.
func WithProjector(prj *LLaMACppRunEstimate) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		o.Projector = prj
	}
}

// WithAdapters sets the adapters estimate usage.
func WithAdapters(adp []LLaMACppRunEstimate) LLaMACppRunEstimateOption {
	return func(o *_LLaMACppRunEstimateOptions) {
		if len(adp) == 0 {
			return
		}
		o.Adapters = adp
	}
}
