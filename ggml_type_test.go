package gguf_parser

import "testing"

// The three types llama.cpp defines above MXFP4. Block sizes come from
// ggml/src/ggml-common.h: QK1_0 128, QK2_0 64, QK_NVFP4 64 with a 16-value
// sub-block, and each block carries an ggml_half scale.
func TestGGMLType_TraitsMatchLlamaCpp(t *testing.T) {
	cases := []struct {
		typ       GGMLType
		id        uint32
		name      string
		blockSize uint64
		typeSize  uint64
	}{
		{GGMLTypeNVFP4, 40, "NVFP4", 64, 36},
		{GGMLTypeQ1_0, 41, "Q1_0", 128, 18},
		{GGMLTypeQ2_0, 42, "Q2_0", 64, 18},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if uint32(c.typ) != c.id {
				t.Fatalf("%s is id %d, llama.cpp assigns %d", c.name, c.typ, c.id)
			}
			if got := c.typ.String(); got != c.name {
				t.Fatalf("String() = %q, want %q", got, c.name)
			}
			tr, ok := c.typ.Trait()
			if !ok {
				t.Fatalf("%s has no trait entry", c.name)
			}
			if tr.BlockSize != c.blockSize || tr.TypeSize != c.typeSize {
				t.Fatalf("%s trait = block %d size %d, want block %d size %d",
					c.name, tr.BlockSize, tr.TypeSize, c.blockSize, c.typeSize)
			}
			if !tr.Quantized {
				t.Fatalf("%s should be quantized", c.name)
			}
		})
	}
}

// _GGMLTypeCount is the boundary the tensor-info reader rejects against, and
// llama.cpp's own bound is "should be in [0, 43)".
func TestGGMLType_CountMatchesLlamaCpp(t *testing.T) {
	if _GGMLTypeCount != 43 {
		t.Fatalf("_GGMLTypeCount = %d, llama.cpp ggml_type ends at 43", _GGMLTypeCount)
	}
}
