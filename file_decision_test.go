package gguf_parser

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// decisionTestFileSpec describes a minimal transformer GGUF file for decision
// model detection tests.
type decisionTestFileSpec struct {
	// architecture is the value of the "general.architecture" metadata.
	architecture string
	// decisionType is the value of the "<architecture>.decision.type" metadata,
	// empty omits the key unless emptyDecisionType is set.
	decisionType string
	// emptyDecisionType declares the "<architecture>.decision.type" key with
	// an empty string value.
	emptyDecisionType bool
	// corruptDecisionTypeValue declares the "<architecture>.decision.type" key
	// with a uint32 value instead of a string, the crafted file the value type
	// guard must survive.
	corruptDecisionTypeValue bool
	// attentionCausal is the value of the "<architecture>.attention.causal"
	// metadata, false for the bidirectional classifier backbones by default.
	attentionCausal bool
	// poolingType is the value of the "<architecture>.pooling_type" metadata,
	// 0 omits the key.
	poolingType uint32
	// clsTensors adds "cls.weight" and "cls.bias" tensor infos, the classifier
	// head the reranking heuristic looks for.
	clsTensors bool
}

// writeDecisionTestFile builds the described GGUF v3 file in a temporary file
// and returns its path. The file holds a complete header and tensor infos but
// no tensor data, which is all the parser and the estimator read.
func writeDecisionTestFile(t *testing.T, spec decisionTestFileSpec) string {
	t.Helper()

	var (
		buf bytes.Buffer
		bo  = binary.LittleEndian
	)
	writeString := func(s string) {
		require.NoError(t, binary.Write(&buf, bo, uint64(len(s))))
		buf.WriteString(s)
	}
	writeUint32 := func(v uint32) {
		require.NoError(t, binary.Write(&buf, bo, v))
	}
	writeUint64 := func(v uint64) {
		require.NoError(t, binary.Write(&buf, bo, v))
	}
	writeStringKV := func(key, value string) {
		writeString(key)
		writeUint32(uint32(GGUFMetadataValueTypeString))
		writeString(value)
	}
	writeUint32KV := func(key string, value uint32) {
		writeString(key)
		writeUint32(uint32(GGUFMetadataValueTypeUint32))
		writeUint32(value)
	}
	writeBoolKV := func(key string, value bool) {
		writeString(key)
		writeUint32(uint32(GGUFMetadataValueTypeBool))
		b := []byte{0}
		if value {
			b[0] = 1
		}
		buf.Write(b)
	}
	writeTensor := func(name string, dims ...uint64) {
		writeString(name)
		writeUint32(uint32(len(dims)))
		for _, dim := range dims {
			writeUint64(dim)
		}
		writeUint32(uint32(GGMLTypeF32))
		writeUint64(0) // Offset, unused by the tests.
	}

	arch := spec.architecture
	type kv struct {
		write func()
	}
	kvs := []kv{
		{func() { writeStringKV("general.architecture", arch) }},
		{func() { writeUint32KV(arch+".context_length", 64) }},
		{func() { writeUint32KV(arch+".embedding_length", 16) }},
		{func() { writeUint32KV(arch+".block_count", 2) }},
		{func() { writeUint32KV(arch+".attention.head_count", 2) }},
		{func() { writeBoolKV(arch+".attention.causal", spec.attentionCausal) }},
	}
	if spec.poolingType != 0 {
		kvs = append(kvs, kv{func() { writeUint32KV(arch+".pooling_type", spec.poolingType) }})
	}
	if spec.decisionType != "" {
		kvs = append(kvs, kv{func() { writeStringKV(arch+".decision.type", spec.decisionType) }})
	}
	if spec.emptyDecisionType {
		kvs = append(kvs, kv{func() { writeStringKV(arch+".decision.type", "") }})
	}
	if spec.corruptDecisionTypeValue {
		kvs = append(kvs, kv{func() { writeUint32KV(arch+".decision.type", 1) }})
	}

	tensorCount := uint64(0)
	if spec.clsTensors {
		tensorCount = 2
	}

	// Header.
	buf.WriteString("GGUF")
	writeUint32(uint32(GGUFVersionV3))
	writeUint64(tensorCount)
	writeUint64(uint64(len(kvs)))

	for _, kv := range kvs {
		kv.write()
	}
	if spec.clsTensors {
		writeTensor("cls.weight", 16, 2)
		writeTensor("cls.bias", 2)
	}

	path := t.TempDir() + "/decision_test.gguf"
	require.NoError(t, os.WriteFile(path, buf.Bytes(), 0o600))
	return path
}

func TestGGUFFile_Architecture_DecisionType(t *testing.T) {
	cases := []struct {
		name             string
		decisionType     string
		expectedDecision string
	}{
		{
			name:             "declared decision model",
			decisionType:     "laya",
			expectedDecision: "laya",
		},
		{
			name:             "unrecognized type still declares a decision model",
			decisionType:     "orgname-modelname",
			expectedDecision: "orgname-modelname",
		},
		{
			name:             "no decision head",
			decisionType:     "",
			expectedDecision: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeDecisionTestFile(t, decisionTestFileSpec{
				architecture: "modernbert",
				decisionType: tc.decisionType,
			})

			f, err := ParseGGUFFile(path)
			require.NoError(t, err)

			a := f.Architecture()
			assert.Equal(t, tc.expectedDecision, a.DecisionType)
		})
	}
}

func TestGGUFFile_Architecture_DecisionTypeWrongValueType(t *testing.T) {
	// A crafted file declaring the key with a non-string value must not
	// panic the parser, it reads as no decision head.
	path := writeDecisionTestFile(t, decisionTestFileSpec{
		architecture:             "modernbert",
		corruptDecisionTypeValue: true,
	})

	f, err := ParseGGUFFile(path)
	require.NoError(t, err)
	assert.Empty(t, f.Architecture().DecisionType)
	assert.False(t, f.EstimateLLaMACppRun().Decision)
}

func TestGGUFFile_Architecture_DecisionTypeJSON(t *testing.T) {
	path := writeDecisionTestFile(t, decisionTestFileSpec{
		architecture: "modernbert",
		decisionType: "openjev",
	})

	f, err := ParseGGUFFile(path)
	require.NoError(t, err)

	b, err := json.Marshal(f.Architecture())
	require.NoError(t, err)
	assert.Contains(t, string(b), `"decisionType":"openjev"`)

	// The field stays omitted for regular models.
	path = writeDecisionTestFile(t, decisionTestFileSpec{architecture: "modernbert"})
	f, err = ParseGGUFFile(path)
	require.NoError(t, err)

	b, err = json.Marshal(f.Architecture())
	require.NoError(t, err)
	assert.NotContains(t, string(b), "decisionType")
}

func TestGGUFFile_DecisionModel_Laya(t *testing.T) {
	// The first-party converted Laya GGUF is a decision model on a modern-bert
	// backbone: it declares "modern-bert.decision.type" and carries the
	// classifier shape the reranking heuristics match. Point TEST_MODEL_PATH
	// at the Laya GGUF to run this test.
	path, ok := os.LookupEnv("TEST_MODEL_PATH")
	if !ok {
		t.Skip("TEST_MODEL_PATH is not set")
		return
	}

	f, err := ParseGGUFFile(path, SkipLargeMetadata())
	require.NoError(t, err)

	a := f.Architecture()
	assert.Equal(t, "modern-bert", a.Architecture)
	assert.Equal(t, "laya", a.DecisionType)

	e := f.EstimateLLaMACppRun()
	assert.True(t, e.Decision)
	assert.False(t, e.Reranking)
}

func TestGGUFFile_EstimateLLaMACppRun_DecisionCausal(t *testing.T) {
	// A causal decision model (e.g. openjev-style label logits) is detected
	// outside the embedding-only branch.
	path := writeDecisionTestFile(t, decisionTestFileSpec{
		architecture:    "llama",
		decisionType:    "openjev",
		attentionCausal: true,
	})

	f, err := ParseGGUFFile(path)
	require.NoError(t, err)

	e := f.EstimateLLaMACppRun()
	assert.True(t, e.Decision)
	assert.False(t, e.EmbeddingOnly)
	assert.False(t, e.Reranking)
}

func TestGGUFFile_EstimateLLaMACppRun_DecisionEmptyMarker(t *testing.T) {
	// A present-but-empty marker still declares a decision model, the
	// detection is presence-only like common_get_decision_type.
	path := writeDecisionTestFile(t, decisionTestFileSpec{
		architecture:      "modernbert",
		emptyDecisionType: true,
		poolingType:       4, // Rank.
		clsTensors:        true,
	})

	f, err := ParseGGUFFile(path)
	require.NoError(t, err)
	assert.Empty(t, f.Architecture().DecisionType)

	e := f.EstimateLLaMACppRun()
	assert.True(t, e.Decision)
	assert.False(t, e.Reranking)
}

func TestGGUFFile_EstimateLLaMACppRun_Decision(t *testing.T) {
	// All cases run on the same non-causal classifier shape: cls head tensors
	// and rank pooling, which reads as a reranker without the decision marker.
	cases := []struct {
		name             string
		decisionType     string
		expectDecision   bool
		expectReranking  bool
		expectEmbedding  bool
		expectSummaryDec bool
	}{
		{
			name:             "decision marker outranks the reranking heuristics",
			decisionType:     "laya",
			expectDecision:   true,
			expectReranking:  false,
			expectEmbedding:  true,
			expectSummaryDec: true,
		},
		{
			name:             "no decision marker falls back to reranking",
			decisionType:     "",
			expectDecision:   false,
			expectReranking:  true,
			expectEmbedding:  true,
			expectSummaryDec: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeDecisionTestFile(t, decisionTestFileSpec{
				architecture: "modernbert",
				decisionType: tc.decisionType,
				poolingType:  4, // Rank.
				clsTensors:   true,
			})

			f, err := ParseGGUFFile(path)
			require.NoError(t, err)

			e := f.EstimateLLaMACppRun()
			assert.Equal(t, tc.expectDecision, e.Decision)
			assert.Equal(t, tc.expectReranking, e.Reranking)
			assert.Equal(t, tc.expectEmbedding, e.EmbeddingOnly)

			es := e.Summarize(true, 0, 0)
			assert.Equal(t, tc.expectSummaryDec, es.Decision)
			assert.Equal(t, tc.expectReranking, es.Reranking)
		})
	}
}
