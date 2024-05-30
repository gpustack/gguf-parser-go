package gguf_parser

import (
	"slices"
)

type (
	_GGUFEstimateOptions struct {
		ContextSize    *int32
		CacheKeyType   *GGMLType
		CacheValueType *GGMLType
		OffloadLayers  *uint64
	}
	GGUFEstimateOption func(*_GGUFEstimateOptions)
)

// WithContextSize sets the context size for the estimate.
func WithContextSize(size int32) GGUFEstimateOption {
	return func(o *_GGUFEstimateOptions) {
		if size <= 0 {
			return
		}
		o.ContextSize = &size
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
func WithCacheKeyType(t GGMLType) GGUFEstimateOption {
	return func(o *_GGUFEstimateOptions) {
		if slices.Contains(_GGUFEstimateCacheTypeAllowList, t) {
			o.CacheKeyType = &t
		}
	}
}

// WithCacheValueType sets the cache value type for the estimate.
func WithCacheValueType(t GGMLType) GGUFEstimateOption {
	return func(o *_GGUFEstimateOptions) {
		if slices.Contains(_GGUFEstimateCacheTypeAllowList, t) {
			o.CacheValueType = &t
		}
	}
}

// WithOffloadLayers sets the number of layers to offload.
func WithOffloadLayers(layers uint64) GGUFEstimateOption {
	return func(o *_GGUFEstimateOptions) {
		if layers <= 0 {
			return
		}
		o.OffloadLayers = &layers
	}
}
