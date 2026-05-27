package gguf_parser

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
)

func TestParseGGUFFile(t *testing.T) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		t.Skip("TEST_MODEL_PATH is not set")
		return
	}

	// Slow read.
	{
		f, err := ParseGGUFFile(mp)
		if err != nil {
			t.Fatal(err)
			return
		}
		s := spew.ConfigState{
			Indent:   "  ",
			MaxDepth: 5, // Avoid console overflow.
		}
		t.Log("\n", s.Sdump(f), "\n")
	}

	// Fast read.
	{
		f, err := ParseGGUFFile(mp, SkipLargeMetadata(), UseMMap())
		if err != nil {
			t.Fatal(err)
			return
		}
		t.Log("\n", spew.Sdump(f), "\n")
	}
}

func BenchmarkParseGGUFFileMMap(b *testing.B) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		b.Skip("TEST_MODEL_PATH is not set")
		return
	}

	b.ReportAllocs()

	b.ResetTimer()
	b.Run("Normal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFile(mp)
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})

	b.ResetTimer()
	b.Run("UseMMap", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFile(mp, UseMMap())
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})
}

func BenchmarkParseGGUFFileSkipLargeMetadata(b *testing.B) {
	mp, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		b.Skip("TEST_MODEL_PATH is not set")
		return
	}

	b.ReportAllocs()

	b.ResetTimer()
	b.Run("Normal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFile(mp, UseMMap())
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})

	b.ResetTimer()
	b.Run("SkipLargeMetadata", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFile(mp, SkipLargeMetadata(), UseMMap())
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})
}

func TestParseGGUFFileRemote(t *testing.T) {
	const u = "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF" +
		"/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf"

	ctx := context.Background()

	// Slow read.
	{
		f, err := ParseGGUFFileRemote(ctx, u, UseDebug())
		if err != nil {
			t.Fatal(err)
			return
		}
		s := spew.ConfigState{
			Indent:   "  ",
			MaxDepth: 5, // Avoid console overflow.
		}
		t.Log("\n", s.Sdump(f), "\n")
	}

	// Fast read.
	{
		f, err := ParseGGUFFileRemote(ctx, u, UseDebug(), SkipLargeMetadata())
		if err != nil {
			t.Fatal(err)
			return
		}
		t.Log("\n", spew.Sdump(f), "\n")
	}
}

func BenchmarkParseGGUFFileRemoteWithBufferSize(b *testing.B) {
	const u = "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF" +
		"/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf"

	ctx := context.Background()

	b.ReportAllocs()

	b.ResetTimer()
	b.Run("256KibBuffer", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFileRemote(ctx, u, SkipLargeMetadata(), UseBufferSize(256*1024))
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})

	b.ResetTimer()
	b.Run("1MibBuffer", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFileRemote(ctx, u, SkipLargeMetadata(), UseBufferSize(1024*1024))
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})

	b.ResetTimer()
	b.Run("4MibBuffer", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ParseGGUFFileRemote(ctx, u, SkipLargeMetadata(), UseBufferSize(4*1024*1024))
			if err != nil {
				b.Fatal(err)
				return
			}
		}
	})
}

func TestParseGGUFFileFromHuggingFace(t *testing.T) {
	ctx := context.Background()

	cases := [][2]string{
		{
			"TheBloke/Llama-2-13B-chat-GGUF",
			"llama-2-13b-chat.Q8_0.gguf",
		},
		{
			"lmstudio-community/Yi-1.5-9B-Chat-GGUF",
			"Yi-1.5-9B-Chat-Q5_K_M.gguf",
		},
		{
			"bartowski/gemma-2-9b-it-GGUF",
			"gemma-2-9b-it-Q3_K_M.gguf",
		},
	}
	for _, tc := range cases {
		t.Run(tc[0]+"/"+tc[1], func(t *testing.T) {
			f, err := ParseGGUFFileFromHuggingFace(ctx, tc[0], tc[1], SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}
			t.Log("\n", spew.Sdump(f), "\n")
		})
	}
}

func TestParseGGUFFileFromModelScope(t *testing.T) {
	ctx := context.Background()

	cases := [][2]string{
		{
			"qwen/Qwen1.5-0.5B-Chat-GGUF",
			"qwen1_5-0_5b-chat-q5_k_m.gguf",
		},
		{
			"HIT-SCIR/huozi3-gguf",
			"huozi3-q2_k.gguf",
		},
		{
			"shaowenchen/chinese-alpaca-2-13b-16k-gguf",
			"chinese-alpaca-2-13b-16k.Q5_K.gguf",
		},
	}
	for _, tc := range cases {
		t.Run(tc[0]+"/"+tc[1], func(t *testing.T) {
			f, err := ParseGGUFFileFromModelScope(ctx, tc[0], tc[1], SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}
			t.Log("\n", spew.Sdump(f), "\n")
		})
	}
}

func TestParseGGUFFileFromOllama(t *testing.T) {
	ctx := context.Background()

	cases := []string{
		"gemma2",
		"llama3.1",
		"qwen2:72b-instruct-q3_K_M",
	}
	for _, tc := range cases {
		t.Run(tc, func(t *testing.T) {
			start := time.Now()
			f, err := ParseGGUFFileFromOllama(ctx, tc, SkipLargeMetadata())
			if err != nil {
				t.Fatal(err)
				return
			}
			t.Logf("cost: %v\n", time.Since(start))
			t.Log("\n", spew.Sdump(f), "\n")
		})
	}
}

// FuzzParseGGUFFile writes the fuzz input to a temp file and calls ParseGGUFFile.
// Any panic during parsing will be reported by the fuzzing harness.
func FuzzParseGGUFFile(f *testing.F) {
	buf := new(bytes.Buffer)
	bo := binary.LittleEndian

	for _, v := range []GGUFMagic{GGUFMagicGGML, GGUFMagicGGMF, GGUFMagicGGJT, GGUFMagicGGUFLe, GGUFMagicGGUFBe} {
		_ = binary.Write(buf, bo, uint32(v))
		f.Add(buf.Bytes())
		buf.Reset()
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		tmp, err := os.CreateTemp("", "gguf_fuzz_*.gguf")
		if err != nil {
			t.Fatalf("create tmp: %v", err)
		}
		defer os.Remove(tmp.Name())

		if _, err := tmp.Write(data); err != nil {
			t.Fatalf("write tmp: %v", err)
		}
		if err := tmp.Close(); err != nil {
			t.Fatalf("close tmp: %v", err)
		}

		// Call the public ParseGGUFFile which exercises parseGGUFFile.
		_, _ = ParseGGUFFile(tmp.Name())
	})
}

func TestParseGGUFFileWithFuzzInput(t *testing.T) {
	// Use the fuzz-generated data
	// data := []byte("GGUF\x00\x00\x00\x030000000000000000")
	data := []byte("FUGG\x00\x00\x00\x00GG>?\x00\x00\x00\x000000")

	// Create temp file
	tmpFile, err := os.CreateTemp("", "fuzz_test_gguf_*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	_, err = tmpFile.Write(data)
	if err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()

	// Parse should return error (since it's invalid or triggers the check)
	_, err = ParseGGUFFile(tmpFile.Name())
	if err == nil {
		t.Error("expected error for fuzz-generated data")
	} else {
		t.Logf("got expected error: %v", err)
	}
}

// Regression tests for CWE-190 (integer overflow) hardening — analog of
// llama.cpp GHSA-vgg9-87g3-85w8. A crafted GGUF with tensor dimensions like
// [0xFFFFFFFFFFFFFFFF, 2, 1, 1] must not silently wrap to a tiny size.

func TestElementsOverflowPanics(t *testing.T) {
	ti := GGUFTensorInfo{
		Name:        "overflow",
		NDimensions: 2,
		Dimensions:  []uint64{math.MaxUint64, 2},
		Type:        GGMLTypeF32,
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on Elements overflow, got none")
		}
	}()
	_ = ti.Elements()
}

func TestBytesOverflowPanics(t *testing.T) {
	ti := GGUFTensorInfo{
		Name:        "overflow",
		NDimensions: 2,
		Dimensions:  []uint64{math.MaxUint64, 2},
		Type:        GGMLTypeF32,
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on Bytes overflow, got none")
		}
	}()
	_ = ti.Bytes()
}

func TestElementsAndBytesValid(t *testing.T) {
	ti := GGUFTensorInfo{
		Name:        "ok",
		NDimensions: 2,
		Dimensions:  []uint64{4, 8},
		Type:        GGMLTypeF32, // TypeSize=4, BlockSize=1
	}
	if got, want := ti.Elements(), uint64(32); got != want {
		t.Fatalf("Elements: got %d, want %d", got, want)
	}
	// nb[0]=4, nb[1]=16; ret = 4 + (4-1)*4 + (8-1)*16 = 4 + 12 + 112 = 128
	if got, want := ti.Bytes(), uint64(128); got != want {
		t.Fatalf("Bytes: got %d, want %d", got, want)
	}
}

func TestMulU64Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on mulU64 overflow")
		}
	}()
	_ = mulU64(math.MaxUint64, 2, "test")
}

func TestAddU64Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on addU64 overflow")
		}
	}()
	_ = addU64(math.MaxUint64, 1, "test")
}

func TestSafeSeekDeltaOverflow(t *testing.T) {
	cases := []struct {
		name   string
		count  uint64
		size   uint64
		wantOK bool
	}{
		{"zero", 0, 8, true},
		{"small", 1024, 8, true},
		{"maxInt64 boundary", math.MaxInt64, 1, true},
		{"just over int64", math.MaxInt64 + 1, 1, false},
		{"uint64-product overflow", math.MaxUint64, 2, false},
		{"len times size overflow", 1 << 62, 8, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := safeSeekDelta(c.count, c.size, c.name)
			if (err == nil) != c.wantOK {
				t.Fatalf("safeSeekDelta(%d,%d) err=%v, wantOK=%v", c.count, c.size, err, c.wantOK)
			}
		})
	}
}

func TestSkipReadingStringRejectsHugeLength(t *testing.T) {
	// Length = math.MaxUint64; int64(l) would have wrapped to -1 and seek
	// backwards. Expect SkipReadingString to surface a parse error instead.
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(math.MaxUint64))
	rd := _GGUFReader{
		v:  GGUFVersionV3,
		bo: binary.LittleEndian,
		f:  bytes.NewReader(buf.Bytes()),
	}
	if err := rd.SkipReadingString(); err == nil {
		t.Fatal("expected error skipping a string with length > MaxInt64")
	}
}

func TestReadArraySkipRejectsOverflowingLen(t *testing.T) {
	// Build the minimal byte stream ReadArray expects: u32 element type +
	// u64 length. Use Uint16 (size 2) with v.Len > MaxInt64/2 so the
	// pre-fix code (int64(v.Len)*2) would have wrapped negative.
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint32(GGUFMetadataValueTypeUint16))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(math.MaxUint64))

	rd := _GGUFReader{
		v:  GGUFVersionV3,
		o:  _GGUFReadOptions{SkipLargeMetadata: true},
		bo: binary.LittleEndian,
		f:  bytes.NewReader(buf.Bytes()),
	}
	// Use a key that does NOT match the always-load suffixes, so we enter
	// the seek-skip branch.
	if _, err := rd.ReadArray("test.array"); err == nil {
		t.Fatal("expected error reading array with overflowing length")
	}
}

func TestTensorInfoReaderRejectsBadNDimensions(t *testing.T) {
	cases := []struct {
		name        string
		nDimensions uint32
	}{
		{"zero dimensions", 0},
		{"too many dimensions", 5},
		{"max uint32", math.MaxUint32},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var buf bytes.Buffer
			// name: u64 length=1, body "x"
			_ = binary.Write(&buf, binary.LittleEndian, uint64(1))
			buf.WriteByte('x')
			// n_dimensions
			_ = binary.Write(&buf, binary.LittleEndian, c.nDimensions)

			rd := _GGUFTensorInfoReader{
				_GGUFReader: _GGUFReader{
					v:  GGUFVersionV3,
					bo: binary.LittleEndian,
					f:  bytes.NewReader(buf.Bytes()),
				},
			}
			if _, err := rd.Read(); err == nil {
				t.Fatalf("expected error for NDimensions=%d", c.nDimensions)
			}
		})
	}
}
