package gguf_parser

import (
	"slices"
)

type (
	_LLaMACppUsageEstimateOptions struct {
		ContextSize    *int32
		BatchSize      *int32
		ParallelSize   *int32
		CacheKeyType   *GGMLType
		CacheValueType *GGMLType
		OffloadLayers  *uint64
		FlashAttention bool
	}
	LLaMACppUsageEstimateOption func(*_LLaMACppUsageEstimateOptions)
)

// WithContextSize sets the context size for the estimate.
func WithContextSize(size int32) LLaMACppUsageEstimateOption {
	return func(o *_LLaMACppUsageEstimateOptions) {
		if size <= 0 {
			return
		}
		o.ContextSize = &size
	}
}

// WithBatchSize sets the physical batch size for the estimate.
func WithBatchSize(size int32) LLaMACppUsageEstimateOption {
	return func(o *_LLaMACppUsageEstimateOptions) {
		if size <= 0 {
			return
		}
		o.BatchSize = &size
	}
}

// WithParallelSize sets the (decoding sequences) parallel size for the estimate.
func WithParallelSize(size int32) LLaMACppUsageEstimateOption {
	return func(o *_LLaMACppUsageEstimateOptions) {
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
func WithCacheKeyType(t GGMLType) LLaMACppUsageEstimateOption {
	return func(o *_LLaMACppUsageEstimateOptions) {
		if slices.Contains(_GGUFEstimateCacheTypeAllowList, t) {
			o.CacheKeyType = &t
		}
	}
}

// WithCacheValueType sets the cache value type for the estimate.
func WithCacheValueType(t GGMLType) LLaMACppUsageEstimateOption {
	return func(o *_LLaMACppUsageEstimateOptions) {
		if slices.Contains(_GGUFEstimateCacheTypeAllowList, t) {
			o.CacheValueType = &t
		}
	}
}

// WithOffloadLayers sets the number of layers to offload.
func WithOffloadLayers(layers uint64) LLaMACppUsageEstimateOption {
	return func(o *_LLaMACppUsageEstimateOptions) {
		o.OffloadLayers = &layers
	}
}

// WithFlashAttention sets the flash attention flag.
func WithFlashAttention() LLaMACppUsageEstimateOption {
	return func(o *_LLaMACppUsageEstimateOptions) {
		o.FlashAttention = true
	}
}
