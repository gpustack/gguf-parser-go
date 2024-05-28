package gguf_parser

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"github.com/dustin/go-humanize"
	"golang.org/x/exp/constraints"

	"github.com/thxcode/gguf-parser-go/util/bytex"
	"github.com/thxcode/gguf-parser-go/util/funcx"
	"github.com/thxcode/gguf-parser-go/util/httpx"
	"github.com/thxcode/gguf-parser-go/util/osx"
)

// GGUFFile represents a GGUF file,
// see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#file-structure.
//
// Compared with the complete GGUF file,
// this structure lacks the tensor data part.
type GGUFFile struct {
	/* Basic */

	// Header is the header of the GGUF file.
	Header GGUFHeader `json:"header"`
	// TensorInfos are the tensor infos of the GGUF file,
	// the size of TensorInfos is equal to `Header.TensorCount`.
	//
	// TensorInfos may be empty if read approximately.
	TensorInfos GGUFTensorInfos `json:"tensorInfos,omitempty"`
	// Padding is the padding size of the GGUF file,
	// which is used to split Header and TensorInfos from tensor data.
	Padding int64 `json:"padding"`
	// TensorDataStartOffset is the offset in bytes of the tensor data in this file.
	//
	// The offset is the start of the file.
	TensorDataStartOffset int64 `json:"tensorDataStartOffset"`

	/* Appendix */

	// ModelSize is the size of the model when loading.
	ModelSize GGUFBytesScalar `json:"modelSize"`
	// ModelParameters is the number of the model parameters.
	ModelParameters GGUFParametersScalar `json:"modelParameters"`
	// ModelBitsPerWeight is the bits per weight of the model,
	// which describes how many bits are used to store a weight,
	// higher is better.
	ModelBitsPerWeight GGUFBitsPerWeightScalar `json:"modelBitsPerWeight"`
}

// Types for scalar.
type (
	// GGUFBytesScalar is the scalar for bytes.
	GGUFBytesScalar uint64

	// GGUFParametersScalar is the scalar for parameters.
	GGUFParametersScalar uint64

	// GGUFBitsPerWeightScalar is the scalar for bits per weight.
	GGUFBitsPerWeightScalar float64
)

// GGUFMagic is a magic number of GGUF file,
// see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#historical-state-of-affairs.
type GGUFMagic uint32

// GGUFMagic constants.
const (
	GGUFMagicGGML   GGUFMagic = 0x67676d6c
	GGUFMagicGGMF   GGUFMagic = 0x67676d66
	GGUFMagicGGJT   GGUFMagic = 0x67676a74
	GGUFMagicGGUFLe GGUFMagic = 0x46554747 // GGUF
	GGUFMagicGGUFBe GGUFMagic = 0x47475546 // GGUF
)

// GGUFVersion is a version of GGUF file format,
// see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#version-history.
type GGUFVersion uint32

// GGUFVersion constants.
const (
	GGUFVersionV1 GGUFVersion = iota + 1
	GGUFVersionV2
	GGUFVersionV3
)

// GGUFHeader represents the header of a GGUF file.
type GGUFHeader struct {
	// Magic is a magic number that announces that this is a GGUF file.
	Magic GGUFMagic `json:"magic"`
	// Version is a version of the GGUF file format.
	Version GGUFVersion `json:"version"`
	// TensorCount is the number of tensors in the file.
	TensorCount uint64 `json:"tensorCount"`
	// MetadataKVCount is the number of key-value pairs in the metadata.
	MetadataKVCount uint64 `json:"metadataKVCount"`
	// MetadataKV are the key-value pairs in the metadata,
	MetadataKV GGUFMetadataKVs `json:"metadataKV"`
}

// GGUFMetadataValueType is a type of GGUF metadata value,
// see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#file-structure.
type GGUFMetadataValueType uint32

// GGUFMetadataValueType constants.
const (
	GGUFMetadataValueTypeUint8 GGUFMetadataValueType = iota
	GGUFMetadataValueTypeInt8
	GGUFMetadataValueTypeUint16
	GGUFMetadataValueTypeInt16
	GGUFMetadataValueTypeUint32
	GGUFMetadataValueTypeInt32
	GGUFMetadataValueTypeFloat32
	GGUFMetadataValueTypeBool
	GGUFMetadataValueTypeString
	GGUFMetadataValueTypeArray
	GGUFMetadataValueTypeUint64
	GGUFMetadataValueTypeInt64
	GGUFMetadataValueTypeFloat64
	_GGUFMetadataValueTypeCount // Unknown
)

// Types for GGUFMetadataKV.
type (
	// GGUFMetadataKV is a key-value pair in the metadata of a GGUF file.
	GGUFMetadataKV struct {
		// Key is the key of the metadata key-value pair,
		// which is no larger than 64 bytes long.
		Key string `json:"key"`
		// ValueType is the type of the metadata value.
		ValueType GGUFMetadataValueType `json:"valueType"`
		// Value is the value of the metadata key-value pair.
		Value any `json:"value"`
	}

	// GGUFMetadataKVArrayValue is a value of a GGUFMetadataKV with type GGUFMetadataValueTypeArray.
	GGUFMetadataKVArrayValue struct {
		/* Basic */

		// Type is the type of the array item.
		Type GGUFMetadataValueType `json:"type"`
		// Len is the length of the array.
		Len uint64 `json:"len"`
		// Array holds all array items.
		//
		// Array may be empty if read approximately.
		Array []any `json:"array,omitempty"`

		/* Appendix */

		// StartOffset is the offset in bytes of the GGUFMetadataKVArrayValue in the GGUFFile file.
		//
		// The offset is the start of the file.
		StartOffset int64 `json:"startOffset"`
	}

	// GGUFMetadataKVs is a list of GGUFMetadataKV.
	GGUFMetadataKVs []GGUFMetadataKV
)

// Types for GGMLType.
type (
	// GGMLType is a type of GGML tensor,
	// see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#file-structure.
	GGMLType uint32

	// GGMLTypeTrait holds the trait of a GGMLType,
	// see https://github.com/ggerganov/ggml/blob/0cbb7c0e053f5419cfbebb46fbf4d4ed60182cf5/src/ggml.c#L564-L918.
	GGMLTypeTrait struct {
		BlockSize uint64 // Original is int, in order to reduce conversion, here we use uint64.
		TypeSize  uint64 // Original is uint32, in order to reduce conversion, here we use uint64.
		Quantized bool
	}
)

// GGMLType constants.
//
// GGMLTypeQ4_2, GGMLTypeQ4_3 are deprecated.
const (
	GGMLTypeF32 GGMLType = iota
	GGMLTypeF16
	GGMLTypeQ4_0
	GGMLTypeQ4_1
	GGMLTypeQ4_2
	GGMLTypeQ4_3
	GGMLTypeQ5_0
	GGMLTypeQ5_1
	GGMLTypeQ8_0
	GGMLTypeQ8_1
	GGMLTypeQ2_K
	GGMLTypeQ3_K
	GGMLTypeQ4_K
	GGMLTypeQ5_K
	GGMLTypeQ6_K
	GGMLTypeQ8_K
	GGMLTypeIQ2_XXS
	GGMLTypeIQ2_XS
	GGMLTypeIQ3_XXS
	GGMLTypeIQ1_S
	GGMLTypeIQ4_NL
	GGMLTypeIQ3_S
	GGMLTypeIQ2_S
	GGMLTypeIQ4_XS
	GGMLTypeI8
	GGMLTypeI16
	GGMLTypeI32
	GGMLTypeI64
	GGMLTypeF64
	GGMLTypeIQ1_M
	GGMLTypeBF16
	_GGMLTypeCount // Unknown
)

// Sizes for GGML constant.
const (
	// GGMLTensorSize is the size of a GGML tensor in bytes,
	// see https://github.com/ggerganov/ggml/blob/0cbb7c0e053f5419cfbebb46fbf4d4ed60182cf5/include/ggml/ggml.h#L606.
	GGMLTensorSize = 368

	// GGMLObjectSize is the size of a GGML object in bytes,
	// see https://github.com/ggerganov/ggml/blob/a10a8b880c059b3b29356eb9a9f8df72f03cdb6a/include/ggml/ggml.h#L563.
	GGMLObjectSize = 32
)

// Types for GGUFTensorInfo.
type (
	// GGUFTensorInfo represents a tensor info in a GGUF file.
	GGUFTensorInfo struct {
		/* Basic */

		// Name is the name of the tensor,
		// which is no larger than 64 bytes long.
		Name string `json:"name"`
		// NDimensions is the number of dimensions of the tensor.
		NDimensions uint32 `json:"nDimensions"`
		// Dimensions is the dimensions of the tensor,
		// the length is NDimensions.
		Dimensions []uint64 `json:"dimensions"`
		// Type is the type of the tensor.
		Type GGMLType `json:"type"`
		// Offset is the offset in bytes of the tensor's data in this file.
		//
		// The offset is relative to tensor data, not to the start of the file.
		Offset uint64 `json:"offset"`

		/* Appendix */

		// StartOffset is the offset in bytes of the GGUFTensorInfo in the GGUFFile file.
		//
		// The offset is the start of the file.
		StartOffset int64 `json:"startOffset"`
	}

	// GGUFTensorInfos is a list of GGUFTensorInfo.
	GGUFTensorInfos []GGUFTensorInfo
)

var ErrGGUFFileInvalidFormat = errors.New("invalid GGUF format")

// ParseGGUFFile parses a GGUF file from the local given path,
// and returns the GGUFFile, or an error if any.
func ParseGGUFFile(path string, opts ...GGUFReadOption) (*GGUFFile, error) {
	var o _GGUFReadOptions
	for _, opt := range opts {
		opt(&o)
	}

	var (
		f io.ReadSeeker
		s int64
	)
	if o.MMap {
		mf, err := osx.OpenMmapFile(path)
		if err != nil {
			return nil, fmt.Errorf("open mmap file: %w", err)
		}
		defer osx.Close(mf)
		f = io.NewSectionReader(mf, 0, mf.Len())
		s = mf.Len()
	} else {
		ff, err := osx.Open(path)
		if err != nil {
			return nil, fmt.Errorf("open file: %w", err)
		}
		defer osx.Close(ff)
		f = ff
		s = funcx.MustNoError(ff.Stat()).Size()
	}

	return parseGGUFFile(s, f, o)
}

// ParseGGUFFileRemote parses a GGUF file from a remote URL,
// and returns a GGUFFile, or an error if any.
func ParseGGUFFileRemote(ctx context.Context, url string, opts ...GGUFReadOption) (*GGUFFile, error) {
	var o _GGUFReadOptions
	for _, opt := range opts {
		opt(&o)
	}

	cli := httpx.Client(
		httpx.ClientOptions().
			WithUserAgent("gguf-parser-go").
			If(o.Debug, func(x *httpx.ClientOption) *httpx.ClientOption {
				return x.WithDebug()
			}).
			WithTimeout(0).
			WithTransport(
				httpx.TransportOptions().
					WithoutKeepalive().
					TimeoutForDial(5*time.Second).
					TimeoutForTLSHandshake(5*time.Second).
					TimeoutForResponseHeader(5*time.Second).
					If(o.SkipProxy, func(x *httpx.TransportOption) *httpx.TransportOption {
						return x.WithoutProxy()
					}).
					If(o.ProxyURL != nil, func(x *httpx.TransportOption) *httpx.TransportOption {
						return x.WithProxy(http.ProxyURL(o.ProxyURL))
					}).
					If(o.SkipTLSVerification, func(x *httpx.TransportOption) *httpx.TransportOption {
						return x.WithoutInsecureVerify()
					})))

	var (
		f io.ReadSeeker
		s int64
	)
	{
		req, err := httpx.NewGetRequestWithContext(ctx, url)
		if err != nil {
			return nil, fmt.Errorf("new request: %w", err)
		}

		var sf *httpx.SeekerFile
		if o.BufferSize > 0 {
			sf, err = httpx.OpenSeekerFileWithSize(cli, req, o.BufferSize, 0)
		} else {
			sf, err = httpx.OpenSeekerFile(cli, req)
		}
		if err != nil {
			return nil, fmt.Errorf("open http file: %w", err)
		}
		defer osx.Close(sf)
		f = io.NewSectionReader(sf, 0, sf.Len())
		s = sf.Len()
	}

	return parseGGUFFile(s, f, o)
}

// ParseGGUFFileFromHuggingFace parses a GGUF file from Hugging Face,
// and returns a GGUFFile, or an error if any.
func ParseGGUFFileFromHuggingFace(ctx context.Context, repo, model string, opts ...GGUFReadOption) (*GGUFFile, error) {
	return ParseGGUFFileRemote(ctx, fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, model), opts...)
}

func parseGGUFFile(s int64, f io.ReadSeeker, o _GGUFReadOptions) (_ *GGUFFile, err error) {
	var gf GGUFFile
	var bo binary.ByteOrder = binary.LittleEndian

	// magic
	if err = binary.Read(f, bo, &gf.Header.Magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	switch gf.Header.Magic {
	default:
		return nil, ErrGGUFFileInvalidFormat
	case GGUFMagicGGML, GGUFMagicGGMF, GGUFMagicGGJT:
		return nil, fmt.Errorf("unsupported format: %s", gf.Header.Magic)
	case GGUFMagicGGUFLe:
	case GGUFMagicGGUFBe:
		bo = binary.BigEndian
	}

	// version
	if err = binary.Read(f, bo, &gf.Header.Version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}

	rd := _GGUFReader{v: gf.Header.Version, o: o, f: f, bo: bo}

	// tensor count
	if gf.Header.Version <= GGUFVersionV1 {
		gf.Header.TensorCount, err = rd.ReadUint64FromUint32()
	} else {
		gf.Header.TensorCount, err = rd.ReadUint64()
	}
	if err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}

	// metadata kv count
	if gf.Header.Version <= GGUFVersionV1 {
		gf.Header.MetadataKVCount, err = rd.ReadUint64FromUint32()
	} else {
		gf.Header.MetadataKVCount, err = rd.ReadUint64()
	}
	if err != nil {
		return nil, fmt.Errorf("read metadata kv count: %w", err)
	}

	// metadata kv
	{
		rd := _GGUFMetadataReader{_GGUFReader: rd}
		kvs := make(GGUFMetadataKVs, gf.Header.MetadataKVCount)
		for i := uint64(0); i < gf.Header.MetadataKVCount; i++ {
			kvs[i], err = rd.Read()
			if err != nil {
				return nil, fmt.Errorf("read metadata kv %d: %w", i, err)
			}
		}
		gf.Header.MetadataKV = kvs
	}

	// tensor infos
	{
		rd := _GGUFTensorInfoReader{_GGUFReader: rd}
		if !o.Approximate {
			tis := make(GGUFTensorInfos, gf.Header.TensorCount)
			for i := uint64(0); i < gf.Header.TensorCount; i++ {
				tis[i], err = rd.Read()
				if err != nil {
					return nil, fmt.Errorf("read tensor info %d: %w", i, err)
				}
			}
			gf.TensorInfos = tis
		} else {
			for i := uint64(0); i < gf.Header.TensorCount; i++ {
				_, err = rd.Read()
				if err != nil {
					return nil, fmt.Errorf("read tensor info %d: %w", i, err)
				}
			}
		}
	}

	pds, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("seek padding start: %w", err)
	}

	// padding
	{
		// The global alignment to use, as described above.
		// This can vary to allow for different alignment schemes, but it must be a multiple of 8.
		// Some writers may not write the alignment.
		// If the alignment is not specified, assume it is 32.
		var ag uint32 = 32
		if v, ok := gf.Header.MetadataKV.Get("general.alignment"); ok {
			ag = v.ValueUint32()
		}
		gf.Padding = int64(ag) - (pds % int64(ag))
	}

	// tensor data offset
	gf.TensorDataStartOffset = pds + gf.Padding

	if o.Approximate {
		// size
		gf.ModelSize = GGUFBytesScalar(s - gf.TensorDataStartOffset)
		// parameters
		gf.ModelParameters = gf.guessParameters()
	} else {
		for i := range gf.TensorInfos {
			// size
			gf.ModelSize += GGUFBytesScalar(gf.TensorInfos[i].Bytes())
			// parameters
			gf.ModelParameters += GGUFParametersScalar(gf.TensorInfos[i].Elements())
		}
	}

	// bpw
	if gf.ModelParameters != 0 {
		gf.ModelBitsPerWeight = GGUFBitsPerWeightScalar(float64(gf.ModelSize) * 8 / float64(gf.ModelParameters))
	}

	return &gf, nil
}

// guessParameters guesses the number of parameters,
// which is inspired by https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L3969-L4388.
func (gf *GGUFFile) guessParameters() GGUFParametersScalar {
	const (
		K = 1e3
		M = 1e3 * K
		B = 1e3 * M

		// https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L1718-L1761
		_14M           = 14 * M
		_17M           = 17 * M
		_22M           = 22 * M
		_33M           = 33 * M
		_70M           = 70 * M
		_109M          = 109 * M
		_137M          = 137 * M
		_160M          = 160 * M
		_335M          = 335 * M
		_410M          = 410 * M
		_0_5B          = 0.5 * B
		_1B            = 1 * B
		_1_4B          = 1.4 * B
		_2B            = 2 * B
		_2_8B          = 2.8 * B
		_3B            = 3 * B
		_4B            = 4 * B
		_6_9B          = 6.9 * B
		_7B            = 7 * B
		_8B            = 8 * B
		_12B           = 12 * B
		_13B           = 13 * B
		_14B           = 14 * B
		_15B           = 15 * B
		_20B           = 20 * B
		_30B           = 30 * B
		_34B           = 34 * B
		_35B           = 35 * B
		_40B           = 40 * B
		_65B           = 65 * B
		_70B           = 70 * B
		_314B          = 314 * B
		_SMALL         = 0.1 * B
		_MEDIUM        = 0.4 * B
		_LARGE         = 0.8 * B
		_XL            = 1.5 * B
		_A2_7B         = 14.3 * B // Guess
		_8x7B          = 47 * B   // Guess
		_8x22B         = 141 * B  // Guess
		_16x12B        = 132 * B  // Guess
		_10B_128x3_66B = 480 * B  // Guess
	)

	arch := "llama"
	if v, ok := gf.Header.MetadataKV.Get("general.architecture"); ok {
		arch = v.ValueString()
	}

	var (
		contextLengthKey        = arch + ".context_length"
		embeddingLengthKey      = arch + ".embedding_length"
		blockCountKey           = arch + ".block_count"
		feedForwardLengthKey    = arch + ".feed_forward_length"
		expertCountKey          = arch + ".expert_count"
		attentionHeadCountKey   = arch + ".attention.head_count"
		attentionHeadCountKVKey = arch + ".attention.head_count_kv"
		vocabularyLengthKey     = arch + ".vocab_size" // uint32 maybe
		tokenizerGGMLTokensKey  = "tokenizer.ggml.tokens"
	)
	m, _ := gf.Header.MetadataKV.Index([]string{
		contextLengthKey,
		embeddingLengthKey,
		blockCountKey,
		feedForwardLengthKey,
		expertCountKey,
		attentionHeadCountKey,
		attentionHeadCountKVKey,
		vocabularyLengthKey,
		tokenizerGGMLTokensKey,
	})

	var (
		embeddingLength      uint64
		blockCount           uint64
		feedForwardLength    uint64
		expertCount          uint32
		attentionHeadCount   uint64
		attentionHeadCountKV uint64
		vocabularyLength     uint64
	)
	if v, ok := m[embeddingLengthKey]; ok {
		embeddingLength = ValueNumeric[uint64](v)
	}
	if v, ok := m[blockCountKey]; ok {
		blockCount = ValueNumeric[uint64](v)
	}
	if v, ok := m[feedForwardLengthKey]; ok {
		feedForwardLength = ValueNumeric[uint64](v)
	}
	if v, ok := m[expertCountKey]; ok {
		expertCount = ValueNumeric[uint32](v)
	}
	if v, ok := m[attentionHeadCountKey]; ok {
		attentionHeadCount = ValueNumeric[uint64](v)
	}
	if v, ok := m[attentionHeadCountKVKey]; ok {
		attentionHeadCountKV = ValueNumeric[uint64](v)
	} else {
		attentionHeadCountKV = attentionHeadCount
	}
	if v, ok := m[vocabularyLengthKey]; ok {
		vocabularyLength = ValueNumeric[uint64](v)
	} else if v, ok := m[tokenizerGGMLTokensKey]; ok {
		vocabularyLength = v.ValueArray().Len
	}

	// Try historical statistics,
	// https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L228-L263
	switch arch {
	case "llama":
		if expertCount == 8 {
			switch blockCount {
			case 32:
				return _8x7B
			case 56:
				return _8x22B
			}
		} else {
			switch blockCount {
			case 22:
				return _1B
			case 26:
				return _3B
			case 32:
				if vocabularyLength < 40000 {
					return _7B
				}
				return _8B
			case 40:
				return _13B
			case 48:
				return _34B
			case 60:
				return _30B
			case 80:
				if attentionHeadCount == attentionHeadCountKV {
					return _65B
				}
				return _70B
			}
		}
	case "falcon":
		switch blockCount {
		case 32:
			return _7B
		case 60:
			return _40B
		}
	case "grok":
		if blockCount == 64 {
			return _314B
		}
	case "gpt2":
		switch blockCount {
		case 12:
			return _SMALL
		case 24:
			return _MEDIUM
		case 36:
			return _LARGE
		case 48:
			return _XL
		}
	case "gptj":
	case "gptneox":
		switch blockCount {
		case 6:
			switch feedForwardLength {
			case 512:
				return _14M
			case 2048:
				return _70M
			}
		case 12:
			if feedForwardLength == 3072 {
				return _160M
			}
		case 16:
			if feedForwardLength == 8192 {
				return _1B
			}
		case 24:
			switch feedForwardLength {
			case 4096:
				return _410M
			case 8192:
				return _1_4B
			}
		case 32:
			switch feedForwardLength {
			case 10240:
				return _2_8B
			case 16384:
				return _6_9B
			}
		case 36:
			if feedForwardLength == 20480 {
				return _12B
			}
		case 44:
			if feedForwardLength == 24576 {
				return _20B
			}
		}
	case "mpt":
		switch blockCount {
		case 32:
			return _7B
		case 48:
			return _30B
		}
	case "baichuan":
		switch blockCount {
		case 32:
			return _7B
		case 40:
			return _13B
		}
	case "starcoder":
		switch blockCount {
		case 24:
			return _1B
		case 36:
			return _3B
		case 42:
			return _7B
		case 40:
			return _15B
		}
	case "refact":
		if blockCount == 32 {
			return _1B
		}
	case "bert":
		switch blockCount {
		case 3:
			return _17M
		case 6:
			return _22M
		case 12:
			switch embeddingLength {
			case 384:
				return _33M
			case 768:
				return _109M
			}
		case 24:
			return _335M
		}
	case "nomic-bert":
		if blockCount == 12 && embeddingLength == 768 {
			return _137M
		}
	case "jina-bert-v2":
		switch blockCount {
		case 4:
			return _33M
		case 12:
			return _137M
		}
	case "bloom":
		switch blockCount {
		case 24:
			return _1B
		case 30:
			switch embeddingLength {
			case 2560:
				return _3B
			case 4096:
				return _7B
			}
		}
	case "stablelm":
		switch blockCount {
		case 24:
			return _1B
		case 32:
			return _3B
		case 40:
			return _12B
		}
	case "qwen":
		switch blockCount {
		case 32:
			return _7B
		case 40:
			return _13B
		}
	case "qwen2":
		switch blockCount {
		case 24:
			if embeddingLength == 1024 {
				return _0_5B
			}
			return _1B
		case 32:
			return _7B
		case 40:
			if attentionHeadCount == 20 {
				return _4B
			}
			return _13B
		case 80:
			return _70B
		}
	case "qwen2moe":
		if blockCount == 24 {
			return _A2_7B
		}
	case "phi2":
		switch blockCount {
		case 24:
			return _1B
		case 32:
			return _3B
		}
	case "phi3":
		switch blockCount {
		case 24:
			return _1B
		case 32:
			return _3B
		case 40:
			return _14B
		}
	case "plamo":
		if blockCount == 40 {
			return _13B
		}
	case "codeshell":
		if blockCount == 42 {
			return _SMALL
		}
	case "orion":
		if blockCount == 40 {
			return _14B
		}
	case "internlm2":
		switch blockCount {
		case 32:
			return _7B
		case 48:
			return _20B
		}
	case "minicpm":
		if blockCount == 40 {
			return _2B
		}
	case "gemma":
		switch blockCount {
		case 18:
			return _2B
		case 28:
			return _7B
		}
	case "starcoder2":
		switch blockCount {
		case 30:
			return _3B
		case 32:
			return _7B
		case 40:
			return _15B
		}
	case "mamba":
		switch blockCount {
		case 24:
			if embeddingLength == 768 {
				return _SMALL
			}
		case 48:
			switch embeddingLength {
			case 1024:
				return _MEDIUM
			case 1536:
				return _LARGE
			case 2048:
				return _XL
			}
		case 64:
			if embeddingLength == 2560 {
				return _3B
			}
		}
	case "xverse":
		switch blockCount {
		case 32:
			return _7B
		case 40:
			return _13B
		case 80:
			return _65B
		}
	case "command-r":
		if blockCount == 40 {
			return _35B
		}
	case "dbrx":
		if blockCount == 40 {
			return _16x12B
		}
	case "olmo":
		switch blockCount {
		case 22:
			return _1B
		case 32:
			return _7B
		case 80:
			return _70B
		}
	case "arctic":
		if expertCount == 128 && blockCount == 35 {
			return _10B_128x3_66B
		}
	}

	// Otherwise, calculate by experience.
	//
	// Let's say, the model is based on Transformer architecture,
	// and use decoder-only.
	//
	// Vocabulary embedding parameter number(VeP), mainly includes the embedding matrix.
	// The embedding matrix shape is [VocabularyLength, EmbeddingLength].
	// So the VeP value is VocabularyLength * EmbeddingLength.
	//
	// Self-Attention parameter number(SaP), includes Wq, Wk, Wv, Wo, and their bias.
	// The all weight matrix shapes are [EmbeddingLength, EmbeddingLength],
	// and the bias shapes are [EmbeddingLength].
	// So the SaP value is 4 * (EmbeddingLength * EmbeddingLength) + 4 * EmbeddingLength.
	//
	// Feed-Forward parameter number(FfP), includes W1, W2, and their bias.
	// The W1 shape is [EmbeddingLength, 4*EmbeddingLength], its bias shape is [4*EmbeddingLength].
	// The W2 shape is [4*EmbeddingLength, EmbeddingLength], its bias shape is [EmbeddingLength].
	// So the FfP value is (EmbeddingLength * 4 * EmbeddingLength) + 4 * EmbeddingLength + (4 * EmbeddingLength * EmbeddingLength) + EmbeddingLength.
	//
	// There are two LayerNorm, one for Self-Attention, and another for Feed-Forward.
	// Layer Normalization parameter number(LnP), includes scale and bias.
	// The scale and bias shapes are [EmbeddingLength].
	// So the LnP value is 2 * (2 * EmbeddingLength).
	//
	// So the total parameters of a decoder-only model can estimate as below.
	// Parameters = BlockCount * (SaP + FfP + LnP) + VeP
	//            = BlockCount * (12 * EmbeddingLength * EmbeddingLength + 13 * EmbeddingLength) + VocabularyLength * EmbeddingLength

	ret := blockCount*(12*embeddingLength*embeddingLength+13*embeddingLength) + vocabularyLength*embeddingLength
	// TODO MoE
	return GGUFParametersScalar(ret)
}

func (s GGUFBytesScalar) String() string {
	return humanize.IBytes(uint64(s))
}

func (s GGUFParametersScalar) String() string {
	switch {
	case s >= 1e15:
		return humanize.CommafWithDigits(float64(s)/1e15, 1) + " Q"
	case s >= 1e12:
		return humanize.CommafWithDigits(float64(s)/1e12, 1) + " T"
	case s >= 1e9:
		return humanize.CommafWithDigits(float64(s)/1e9, 1) + " B"
	case s >= 1e6:
		return humanize.CommafWithDigits(float64(s)/1e6, 1) + " M"
	case s >= 1e3:
		return humanize.CommafWithDigits(float64(s)/1e3, 1) + " K"
	default:
		return strconv.Itoa(int(s))
	}
}

func (s GGUFBitsPerWeightScalar) String() string {
	if s == 0 {
		return "Unknown"
	}
	return strconv.FormatFloat(float64(s), 'f', 2, 64) + " bpw"
}

func (kv GGUFMetadataKV) ValueUint8() uint8 {
	if kv.ValueType != GGUFMetadataValueTypeUint8 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(uint8)
}

func (kv GGUFMetadataKV) ValueInt8() int8 {
	if kv.ValueType != GGUFMetadataValueTypeInt8 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(int8)
}

func (kv GGUFMetadataKV) ValueUint16() uint16 {
	if kv.ValueType != GGUFMetadataValueTypeUint16 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(uint16)
}

func (kv GGUFMetadataKV) ValueInt16() int16 {
	if kv.ValueType != GGUFMetadataValueTypeInt16 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(int16)
}

func (kv GGUFMetadataKV) ValueUint32() uint32 {
	if kv.ValueType != GGUFMetadataValueTypeUint32 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(uint32)
}

func (kv GGUFMetadataKV) ValueInt32() int32 {
	if kv.ValueType != GGUFMetadataValueTypeInt32 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(int32)
}

func (kv GGUFMetadataKV) ValueFloat32() float32 {
	if kv.ValueType != GGUFMetadataValueTypeFloat32 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(float32)
}

func (kv GGUFMetadataKV) ValueBool() bool {
	if kv.ValueType != GGUFMetadataValueTypeBool {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(bool)
}

func (kv GGUFMetadataKV) ValueString() string {
	if kv.ValueType != GGUFMetadataValueTypeString {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(string)
}

func (kv GGUFMetadataKV) ValueArray() GGUFMetadataKVArrayValue {
	if kv.ValueType != GGUFMetadataValueTypeArray {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(GGUFMetadataKVArrayValue)
}

func (kv GGUFMetadataKV) ValueUint64() uint64 {
	if kv.ValueType != GGUFMetadataValueTypeUint64 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(uint64)
}

func (kv GGUFMetadataKV) ValueInt64() int64 {
	if kv.ValueType != GGUFMetadataValueTypeInt64 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(int64)
}

func (kv GGUFMetadataKV) ValueFloat64() float64 {
	if kv.ValueType != GGUFMetadataValueTypeFloat64 {
		panic(fmt.Errorf("invalid type: %v", kv.ValueType))
	}
	return kv.Value.(float64)
}

// ValueNumeric returns the numeric values of the GGUFMetadataKV,
// and panics if the value type is not numeric.
//
// ValueNumeric is a generic function, and the type T must be constraints.Integer or constraints.Float.
//
// Compare to the GGUFMetadataKV's Value* functions,
// ValueNumeric will cast the original value to the target type.
func ValueNumeric[T constraints.Integer | constraints.Float](kv GGUFMetadataKV) T {
	switch kv.ValueType {
	case GGUFMetadataValueTypeUint8:
		return T(kv.Value.(uint8))
	case GGUFMetadataValueTypeInt8:
		return T(kv.Value.(int8))
	case GGUFMetadataValueTypeUint16:
		return T(kv.Value.(int16))
	case GGUFMetadataValueTypeInt16:
		return T(kv.Value.(int16))
	case GGUFMetadataValueTypeUint32:
		return T(kv.Value.(uint32))
	case GGUFMetadataValueTypeInt32:
		return T(kv.Value.(int32))
	case GGUFMetadataValueTypeFloat32:
		return T(kv.Value.(float32))
	case GGUFMetadataValueTypeUint64:
		return T(kv.Value.(uint64))
	case GGUFMetadataValueTypeInt64:
		return T(kv.Value.(int64))
	case GGUFMetadataValueTypeFloat64:
		return T(kv.Value.(float64))
	default:
	}
	panic(fmt.Errorf("invalid type: %v", kv.ValueType))
}

func (av GGUFMetadataKVArrayValue) ValuesUint8() []uint8 {
	if av.Type != GGUFMetadataValueTypeUint8 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]uint8, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(uint8)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesInt8() []int8 {
	if av.Type != GGUFMetadataValueTypeInt8 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]int8, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(int8)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesUint16() []uint16 {
	if av.Type != GGUFMetadataValueTypeUint16 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]uint16, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(uint16)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesInt16() []int16 {
	if av.Type != GGUFMetadataValueTypeInt16 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]int16, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(int16)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesUint32() []uint32 {
	if av.Type != GGUFMetadataValueTypeUint32 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]uint32, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(uint32)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesInt32() []int32 {
	if av.Type != GGUFMetadataValueTypeInt32 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]int32, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(int32)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesFloat32() []float32 {
	if av.Type != GGUFMetadataValueTypeFloat32 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]float32, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(float32)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesBool() []bool {
	if av.Type != GGUFMetadataValueTypeBool {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]bool, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(bool)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesString() []string {
	if av.Type != GGUFMetadataValueTypeString {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]string, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(string)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesArray() []GGUFMetadataKVArrayValue {
	if av.Type != GGUFMetadataValueTypeArray {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]GGUFMetadataKVArrayValue, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(GGUFMetadataKVArrayValue)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesUint64() []uint64 {
	if av.Type != GGUFMetadataValueTypeUint64 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]uint64, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(uint64)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesInt64() []int64 {
	if av.Type != GGUFMetadataValueTypeInt64 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]int64, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(int64)
	}
	return v
}

func (av GGUFMetadataKVArrayValue) ValuesFloat64() []float64 {
	if av.Type != GGUFMetadataValueTypeFloat64 {
		panic(fmt.Errorf("invalid type: %v", av.Type))
	}
	v := make([]float64, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		v[i] = av.Array[i].(float64)
	}
	return v
}

// ValuesNumeric returns the numeric values of the GGUFMetadataKVArrayValue,
// and panics if the value type is not numeric.
//
// ValuesNumeric is a generic function, and the type T must be constraints.Integer or constraints.Float.
//
// Compare to the GGUFMetadataKVArrayValue's Value* functions,
// ValuesNumeric will cast the original value to the target type.
func ValuesNumeric[T constraints.Integer | constraints.Float](av GGUFMetadataKVArrayValue) []T {
	v := make([]T, av.Len)
	for i := uint64(0); i < av.Len; i++ {
		switch av.Type {
		case GGUFMetadataValueTypeUint8:
			v[i] = T(av.Array[i].(uint8))
		case GGUFMetadataValueTypeInt8:
			v[i] = T(av.Array[i].(int8))
		case GGUFMetadataValueTypeUint16:
			v[i] = T(av.Array[i].(uint16))
		case GGUFMetadataValueTypeInt16:
			v[i] = T(av.Array[i].(int16))
		case GGUFMetadataValueTypeUint32:
			v[i] = T(av.Array[i].(uint32))
		case GGUFMetadataValueTypeInt32:
			v[i] = T(av.Array[i].(int32))
		case GGUFMetadataValueTypeFloat32:
			v[i] = T(av.Array[i].(float32))
		case GGUFMetadataValueTypeUint64:
			v[i] = T(av.Array[i].(uint64))
		case GGUFMetadataValueTypeInt64:
			v[i] = T(av.Array[i].(int64))
		case GGUFMetadataValueTypeFloat64:
			v[i] = T(av.Array[i].(float64))
		default:
			panic(fmt.Errorf("invalid type: %v", av.Type))
		}
	}
	return v
}

// HasAll returns true if the GGUFMetadataKVs has all the given keys,
// and false otherwise.
func (kvs GGUFMetadataKVs) HasAll(keys []string) bool {
	ks := make(map[string]struct{}, len(keys))
	for i := range keys {
		ks[keys[i]] = struct{}{}
	}
	for i := range kvs {
		k := kvs[i].Key
		if _, ok := ks[k]; !ok {
			continue
		}
		delete(ks, k)
		if len(ks) == 0 {
			break
		}
	}
	return len(ks) == 0
}

// Get returns the GGUFMetadataKV with the given key,
// and true if found, and false otherwise.
func (kvs GGUFMetadataKVs) Get(key string) (value GGUFMetadataKV, found bool) {
	for i := range kvs {
		if kvs[i].Key == key {
			return kvs[i], true
		}
	}
	return GGUFMetadataKV{}, false
}

// Search returns a list of GGUFMetadataKV with the keys that match the given regex.
func (kvs GGUFMetadataKVs) Search(keyRegex *regexp.Regexp) (values []GGUFMetadataKV) {
	for i := range kvs {
		if keyRegex.MatchString(kvs[i].Key) {
			values = append(values, kvs[i])
		}
	}
	return values
}

// Index returns a map value to the GGUFMetadataKVs with the given keys,
// and the number of keys found.
func (kvs GGUFMetadataKVs) Index(keys []string) (values map[string]GGUFMetadataKV, found int) {
	ks := make(map[string]struct{}, len(keys))
	for i := range keys {
		ks[keys[i]] = struct{}{}
	}
	values = make(map[string]GGUFMetadataKV)
	for i := range kvs {
		if _, ok := ks[kvs[i].Key]; ok {
			values[kvs[i].Key] = kvs[i]
			found++
		}
		if found == len(ks) {
			break
		}
	}
	return values, found
}

// _GGMLTypeTraits is a table of GGMLTypeTrait for GGMLType.
var _GGMLTypeTraits = map[GGMLType]GGMLTypeTrait{
	GGMLTypeF32:     {BlockSize: 1, TypeSize: 4},
	GGMLTypeF16:     {BlockSize: 1, TypeSize: 2},
	GGMLTypeQ4_0:    {BlockSize: 32, TypeSize: 18, Quantized: true},
	GGMLTypeQ4_1:    {BlockSize: 32, TypeSize: 20, Quantized: true},
	GGMLTypeQ4_2:    {BlockSize: 0, TypeSize: 0}, // Deprecated
	GGMLTypeQ4_3:    {BlockSize: 0, TypeSize: 0}, // Deprecated
	GGMLTypeQ5_0:    {BlockSize: 32, TypeSize: 22, Quantized: true},
	GGMLTypeQ5_1:    {BlockSize: 32, TypeSize: 24, Quantized: true},
	GGMLTypeQ8_0:    {BlockSize: 32, TypeSize: 34, Quantized: true},
	GGMLTypeQ8_1:    {BlockSize: 32, TypeSize: 36, Quantized: true},
	GGMLTypeQ2_K:    {BlockSize: 256, TypeSize: 84, Quantized: true},
	GGMLTypeQ3_K:    {BlockSize: 256, TypeSize: 110, Quantized: true},
	GGMLTypeQ4_K:    {BlockSize: 256, TypeSize: 144, Quantized: true},
	GGMLTypeQ5_K:    {BlockSize: 256, TypeSize: 176, Quantized: true},
	GGMLTypeQ6_K:    {BlockSize: 256, TypeSize: 210, Quantized: true},
	GGMLTypeQ8_K:    {BlockSize: 256, TypeSize: 292, Quantized: true},
	GGMLTypeIQ2_XXS: {BlockSize: 256, TypeSize: 66, Quantized: true},
	GGMLTypeIQ2_XS:  {BlockSize: 256, TypeSize: 74, Quantized: true},
	GGMLTypeIQ3_XXS: {BlockSize: 256, TypeSize: 98, Quantized: true},
	GGMLTypeIQ1_S:   {BlockSize: 256, TypeSize: 50, Quantized: true},
	GGMLTypeIQ4_NL:  {BlockSize: 32, TypeSize: 18, Quantized: true},
	GGMLTypeIQ3_S:   {BlockSize: 256, TypeSize: 110, Quantized: true},
	GGMLTypeIQ2_S:   {BlockSize: 256, TypeSize: 82, Quantized: true},
	GGMLTypeIQ4_XS:  {BlockSize: 256, TypeSize: 136, Quantized: true},
	GGMLTypeI8:      {BlockSize: 1, TypeSize: 1},
	GGMLTypeI16:     {BlockSize: 1, TypeSize: 2},
	GGMLTypeI32:     {BlockSize: 1, TypeSize: 4},
	GGMLTypeI64:     {BlockSize: 1, TypeSize: 8},
	GGMLTypeF64:     {BlockSize: 1, TypeSize: 8},
	GGMLTypeIQ1_M:   {BlockSize: 256, TypeSize: 56, Quantized: true},
	GGMLTypeBF16:    {BlockSize: 1, TypeSize: 2},
}

// Trait returns the GGMLTypeTrait of the GGMLType.
func (t GGMLType) Trait() (GGMLTypeTrait, bool) {
	tt, ok := _GGMLTypeTraits[t]
	return tt, ok
}

// RowSizeOf returns the size of the given dimensions according to the GGMLType's GGMLTypeTrait,
// which is inspired by
// https://github.com/ggerganov/ggml/blob/0cbb7c0e053f5419cfbebb46fbf4d4ed60182cf5/src/ggml.c#L3142-L3145.
//
// The index of the given dimensions means the number of dimension,
// i.e. 0 is the first dimension, 1 is the second dimension, and so on.
//
// The value of the item is the number of elements in the corresponding dimension.
func (t GGMLType) RowSizeOf(dimensions []uint64) uint64 {
	if len(dimensions) == 0 {
		panic(errors.New("no dimensions"))
	}

	tt, ok := t.Trait()
	if !ok {
		panic(fmt.Errorf("invalid type: %v", t))
	}

	// https://github.com/ggerganov/ggml/blob/a10a8b880c059b3b29356eb9a9f8df72f03cdb6a/src/ggml.c#L2640-L2643
	ds := tt.TypeSize * dimensions[0] / tt.BlockSize // Row size
	for i := 1; i < len(dimensions); i++ {
		ds *= dimensions[i]
	}
	return ds
}

// Elements returns the number of elements of the GGUFTensorInfo,
// which is inspired by
// https://github.com/ggerganov/ggml/blob/a10a8b880c059b3b29356eb9a9f8df72f03cdb6a/src/ggml.c#L2597-L2601.
func (ti GGUFTensorInfo) Elements() uint64 {
	if ti.NDimensions == 0 {
		panic(errors.New("no dimensions"))
	}

	ret := uint64(1)
	for i := uint32(0); i < ti.NDimensions; i++ {
		ret *= ti.Dimensions[i]
	}
	return ret
}

// Bytes returns the number of bytes of the GGUFTensorInfo,
// which is inspired by
// https://github.com/ggerganov/ggml/blob/a10a8b880c059b3b29356eb9a9f8df72f03cdb6a/src/ggml.c#L2609-L2626.
func (ti GGUFTensorInfo) Bytes() uint64 {
	if ti.NDimensions == 0 {
		panic(errors.New("no dimensions"))
	}

	tt, ok := ti.Type.Trait()
	if !ok {
		panic(fmt.Errorf("invalid type: %v", ti.Type))
	}

	// https://github.com/ggerganov/ggml/blob/a10a8b880c059b3b29356eb9a9f8df72f03cdb6a/src/ggml.c#L3210-L3214
	nb := make([]uint64, 0, ti.NDimensions)
	{
		nb = append(nb, tt.TypeSize)
		nb = append(nb, nb[0]*(ti.Dimensions[0]/tt.BlockSize))
		for i := uint32(2); i < ti.NDimensions; i++ {
			nb = append(nb, nb[i-1]*ti.Dimensions[i-1])
		}
	}

	var ret uint64
	if tt.BlockSize == 1 {
		ret = tt.TypeSize
		for i := uint32(0); i < ti.NDimensions; i++ {
			ret += (ti.Dimensions[i] - 1) * nb[i]
		}
		return ret
	}

	ret = ti.Dimensions[0] * nb[0] / tt.BlockSize
	for i := uint32(1); i < ti.NDimensions; i++ {
		ret += (ti.Dimensions[i] - 1) * nb[i]
	}
	return ret
}

// HasAll returns true if the GGUFTensorInfos has all the given names,
// and false otherwise.
func (tis GGUFTensorInfos) HasAll(names []string) bool {
	ns := make(map[string]struct{}, len(names))
	for i := range names {
		ns[names[i]] = struct{}{}
	}
	for i := range tis {
		n := tis[i].Name
		if _, ok := ns[n]; !ok {
			continue
		}
		delete(ns, n)
		if len(ns) == 0 {
			break
		}
	}
	return len(ns) == 0
}

// Get returns the GGUFTensorInfo with the given name,
// and true if found, and false otherwise.
func (tis GGUFTensorInfos) Get(name string) (info GGUFTensorInfo, found bool) {
	for i := range tis {
		if tis[i].Name == name {
			return tis[i], true
		}
	}
	return GGUFTensorInfo{}, false
}

// Search returns a list of GGUFTensorInfo with the names that match the given regex.
func (tis GGUFTensorInfos) Search(nameRegex *regexp.Regexp) (infos []GGUFTensorInfo) {
	for i := range tis {
		if nameRegex.MatchString(tis[i].Name) {
			infos = append(infos, tis[i])
		}
	}
	return infos
}

// Index returns a map value to the GGUFTensorInfos with the given names,
// and the number of names found.
func (tis GGUFTensorInfos) Index(names []string) (infos map[string]GGUFTensorInfo, found int) {
	ns := make(map[string]struct{}, len(names))
	for i := range names {
		ns[names[i]] = struct{}{}
	}
	infos = make(map[string]GGUFTensorInfo)
	for i := range tis {
		if _, ok := ns[tis[i].Name]; ok {
			infos[tis[i].Name] = tis[i]
			found++
		}
		if found == len(ns) {
			break
		}
	}
	return infos, found
}

type _GGUFReader struct {
	v  GGUFVersion
	o  _GGUFReadOptions
	f  io.ReadSeeker
	bo binary.ByteOrder
}

func (rd _GGUFReader) ReadUint8() (v uint8, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read uint8: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadInt8() (v int8, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read int8: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadUint16() (v uint16, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read uint16: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadInt16() (v int16, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read int16: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadUint32() (v uint32, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read uint32: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadUint64FromUint32() (uint64, error) {
	v, err := rd.ReadUint32()
	return uint64(v), err
}

func (rd _GGUFReader) ReadInt32() (v int32, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read int32: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadFloat32() (v float32, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read float32: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadBool() (v bool, err error) {
	b, err := rd.ReadUint8()
	if err != nil {
		return false, fmt.Errorf("read bool: %w", err)
	}
	return b != 0, nil
}

func (rd _GGUFReader) ReadString() (v string, err error) {
	var l uint64
	if rd.v <= GGUFVersionV1 {
		l, err = rd.ReadUint64FromUint32()
	} else {
		l, err = rd.ReadUint64()
	}
	if err != nil {
		return "", fmt.Errorf("read string length: %w", err)
	}

	b := bytex.GetBytes(l)
	defer bytex.Put(b)
	if _, err = rd.f.Read(b); err != nil {
		return "", fmt.Errorf("read string: %w", err)
	}

	return string(bytes.TrimSpace(b)), nil
}

func (rd _GGUFReader) SkipReadingString() (err error) {
	var l uint64
	if rd.v <= GGUFVersionV1 {
		l, err = rd.ReadUint64FromUint32()
	} else {
		l, err = rd.ReadUint64()
	}
	if err != nil {
		return fmt.Errorf("read string length: %w", err)
	}
	_, err = rd.f.Seek(int64(l), io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("seek string: %w", err)
	}
	return nil
}

func (rd _GGUFReader) ReadArray() (v GGUFMetadataKVArrayValue, err error) {
	v.StartOffset, err = rd.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return v, fmt.Errorf("read array start: %w", err)
	}

	if err = binary.Read(rd.f, rd.bo, &v.Type); err != nil {
		return v, fmt.Errorf("read array item type: %w", err)
	}

	if rd.v <= GGUFVersionV1 {
		v.Len, err = rd.ReadUint64FromUint32()
	} else {
		v.Len, err = rd.ReadUint64()
	}
	if err != nil {
		return v, fmt.Errorf("read array length: %w", err)
	}

	if !rd.o.Approximate {
		v.Array = make([]any, v.Len)
		for i := uint64(0); i < v.Len; i++ {
			v.Array[i], err = rd.ReadValue(v.Type)
			if err != nil {
				return v, fmt.Errorf("read array item %d: %w", i, err)
			}
		}

		return v, nil
	}

	switch v.Type {
	case GGUFMetadataValueTypeUint8, GGUFMetadataValueTypeInt8, GGUFMetadataValueTypeBool:
		_, err = rd.f.Seek(int64(v.Len), io.SeekCurrent)
	case GGUFMetadataValueTypeUint16, GGUFMetadataValueTypeInt16:
		_, err = rd.f.Seek(int64(v.Len)*2, io.SeekCurrent)
	case GGUFMetadataValueTypeUint32, GGUFMetadataValueTypeInt32, GGUFMetadataValueTypeFloat32:
		_, err = rd.f.Seek(int64(v.Len)*4, io.SeekCurrent)
	case GGUFMetadataValueTypeUint64, GGUFMetadataValueTypeInt64, GGUFMetadataValueTypeFloat64:
		_, err = rd.f.Seek(int64(v.Len)*8, io.SeekCurrent)
	case GGUFMetadataValueTypeString:
		for i := uint64(0); i < v.Len; i++ {
			if err = rd.SkipReadingString(); err != nil {
				return v, fmt.Errorf("seek array[string] %d: %w", i, err)
			}
		}
	default:
		// Should not happen.
		panic(fmt.Errorf("invalid type: %v", v.Type))
	}
	if err != nil {
		return v, fmt.Errorf("seek array end: %w", err)
	}

	return v, nil
}

func (rd _GGUFReader) ReadUint64() (v uint64, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read uint64: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadInt64() (v int64, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read int64: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadFloat64() (v float64, err error) {
	err = binary.Read(rd.f, rd.bo, &v)
	if err != nil {
		return 0, fmt.Errorf("read float64: %w", err)
	}
	return v, nil
}

func (rd _GGUFReader) ReadValue(vt GGUFMetadataValueType) (v any, err error) {
	if vt >= _GGUFMetadataValueTypeCount {
		return nil, fmt.Errorf("invalid type: %v", vt)
	}

	switch vt {
	case GGUFMetadataValueTypeUint8:
		v, err = rd.ReadUint8()
	case GGUFMetadataValueTypeInt8:
		v, err = rd.ReadInt8()
	case GGUFMetadataValueTypeUint16:
		v, err = rd.ReadUint16()
	case GGUFMetadataValueTypeInt16:
		v, err = rd.ReadInt16()
	case GGUFMetadataValueTypeUint32:
		v, err = rd.ReadUint32()
	case GGUFMetadataValueTypeInt32:
		v, err = rd.ReadInt32()
	case GGUFMetadataValueTypeFloat32:
		v, err = rd.ReadFloat32()
	case GGUFMetadataValueTypeBool:
		v, err = rd.ReadBool()
	case GGUFMetadataValueTypeString:
		v, err = rd.ReadString()
	case GGUFMetadataValueTypeArray:
		v, err = rd.ReadArray()
	case GGUFMetadataValueTypeUint64:
		v, err = rd.ReadUint64()
	case GGUFMetadataValueTypeInt64:
		v, err = rd.ReadInt64()
	case GGUFMetadataValueTypeFloat64:
		v, err = rd.ReadFloat64()
	default:
		// Should not happen.
		panic(fmt.Errorf("invalid type: %v", vt))
	}
	if err != nil {
		return nil, err
	}
	return v, nil
}

type _GGUFMetadataReader struct {
	_GGUFReader
}

func (rd _GGUFMetadataReader) Read() (kv GGUFMetadataKV, err error) {
	kv.Key, err = rd.ReadString()
	if err != nil {
		return kv, fmt.Errorf("read key: %w", err)
	}

	{
		vt, err := rd.ReadUint32()
		if err != nil {
			return kv, fmt.Errorf("read value type: %w", err)
		}
		kv.ValueType = GGUFMetadataValueType(vt)
		if kv.ValueType >= _GGUFMetadataValueTypeCount {
			return kv, fmt.Errorf("invalid value type: %v", kv.ValueType)
		}
	}

	kv.Value, err = rd.ReadValue(kv.ValueType)
	if err != nil {
		return kv, fmt.Errorf("read %s value: %w", kv.Key, err)
	}

	return kv, nil
}

type _GGUFTensorInfoReader struct {
	_GGUFReader
}

func (rd _GGUFTensorInfoReader) Read() (ti GGUFTensorInfo, err error) {
	ti.StartOffset, err = rd.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return ti, fmt.Errorf("seek tensor info start: %w", err)
	}

	if !rd.o.Approximate {
		ti.Name, err = rd.ReadString()
		if err != nil {
			return ti, fmt.Errorf("read name: %w", err)
		}

		ti.NDimensions, err = rd.ReadUint32()
		if err != nil {
			return ti, fmt.Errorf("read n dimensions: %w", err)
		}

		ti.Dimensions = make([]uint64, ti.NDimensions)
		for i := uint32(0); i < ti.NDimensions; i++ {
			if rd.v <= GGUFVersionV1 {
				ti.Dimensions[i], err = rd.ReadUint64FromUint32()
			} else {
				ti.Dimensions[i], err = rd.ReadUint64()
			}
			if err != nil {
				return ti, fmt.Errorf("read dimension %d: %w", i, err)
			}
		}

		{
			v, err := rd.ReadUint32()
			if err != nil {
				return ti, fmt.Errorf("read type: %w", err)
			}
			ti.Type = GGMLType(v)
			if ti.Type >= _GGMLTypeCount {
				return ti, fmt.Errorf("invalid type: %v", ti.Type)
			}
		}

		ti.Offset, err = rd.ReadUint64()
		if err != nil {
			return ti, fmt.Errorf("read offset: %w", err)
		}

		return ti, nil
	}

	err = rd.SkipReadingString()
	if err != nil {
		return ti, fmt.Errorf("seek name: %w", err)
	}

	nd, err := rd.ReadUint32()
	if err != nil {
		return ti, fmt.Errorf("seek n dimensions: %w", err)
	}

	if rd.v <= GGUFVersionV1 {
		_, err = rd.f.Seek(int64(nd)*4 + /* Dimension */ +4 /* Type */ + 8 /* Offset */, io.SeekCurrent)
	} else {
		_, err = rd.f.Seek(int64(nd)*8 /* Dimension */ +4 /* Type */ +8 /* Offset */, io.SeekCurrent)
	}
	if err != nil {
		return ti, fmt.Errorf("seek dimensions/type/offset: %w", err)
	}

	return ti, nil
}
