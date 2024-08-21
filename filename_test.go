package gguf_parser

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/gpustack/gguf-parser-go/util/ptr"
)

func TestParseGGUFFilename(t *testing.T) {
	cases := []struct {
		given    string
		expected *GGUFFilename
	}{
		{
			given: "Mixtral-8x7B-V0.1-KQ2.gguf",
			expected: &GGUFFilename{
				BaseName:  "Mixtral",
				SizeLabel: "8x7B",
				Version:   "V0.1",
				Encoding:  "KQ2",
			},
		},
		{
			given: "Grok-100B-v1.0-Q4_0-00003-of-00009.gguf",
			expected: &GGUFFilename{
				BaseName:   "Grok",
				SizeLabel:  "100B",
				Version:    "v1.0",
				Encoding:   "Q4_0",
				Shard:      ptr.To(3),
				ShardTotal: ptr.To(9),
			},
		},
		{
			given: "Hermes-2-Pro-Llama-3-8B-F16.gguf",
			expected: &GGUFFilename{
				BaseName:  "Hermes 2 Pro Llama 3",
				SizeLabel: "8B",
				Encoding:  "F16",
			},
		},
		{
			given: "Phi-3-mini-3.8B-ContextLength4k-instruct-v1.0.gguf",
			expected: &GGUFFilename{
				BaseName:  "Phi 3 mini",
				SizeLabel: "3.8B-ContextLength4k",
				FineTune:  "instruct",
				Version:   "v1.0",
			},
		},
		{
			given: "Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00001-of-00018.gguf",
			expected: &GGUFFilename{
				BaseName:   "Meta Llama 3.1",
				SizeLabel:  "405B",
				FineTune:   "Instruct-XelotX",
				Encoding:   "BF16",
				Shard:      ptr.To(1),
				ShardTotal: ptr.To(18),
			},
		},
		{
			given: "qwen2-72b-instruct-q6_k-00001-of-00002.gguf",
			expected: &GGUFFilename{
				BaseName:   "qwen2",
				SizeLabel:  "72b",
				FineTune:   "instruct",
				Encoding:   "q6_k",
				Shard:      ptr.To(1),
				ShardTotal: ptr.To(2),
			},
		},
		{
			given:    "Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00001-of-00009.gguf",
			expected: nil,
		},
		{
			given:    "not-a-known-arrangement.gguf",
			expected: nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.given, func(t *testing.T) {
			actual := ParseGGUFFilename(tc.given)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestGGUFFilenameString(t *testing.T) {
	cases := []struct {
		given    GGUFFilename
		expected string
	}{
		{
			given: GGUFFilename{
				BaseName:  "Mixtral",
				SizeLabel: "8x7B",
				Version:   "v0.1",
				Encoding:  "KQ2",
			},
			expected: "Mixtral-8x7B-v0.1-KQ2.gguf",
		},
		{
			given: GGUFFilename{
				BaseName:   "Grok",
				SizeLabel:  "100B",
				Version:    "v1.0",
				Encoding:   "Q4_0",
				Shard:      ptr.To(3),
				ShardTotal: ptr.To(9),
			},
			expected: "Grok-100B-v1.0-Q4_0-00003-of-00009.gguf",
		},
		{
			given: GGUFFilename{
				BaseName:  "Hermes 2 Pro Llama 3",
				SizeLabel: "8B",
				Encoding:  "F16",
			},
			expected: "Hermes-2-Pro-Llama-3-8B-F16.gguf",
		},
		{
			given: GGUFFilename{
				BaseName:  "Phi 3 mini",
				SizeLabel: "3.8B-ContextLength4k",
				FineTune:  "instruct",
				Version:   "v1.0",
			},
			expected: "Phi-3-mini-3.8B-ContextLength4k-instruct-v1.0.gguf",
		},
		{
			given:    GGUFFilename{},
			expected: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.expected, func(t *testing.T) {
			actual := tc.given.String()
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestIsShardGGUFFilename(t *testing.T) {
	cases := []struct {
		given    string
		expected bool
	}{
		{
			given:    "qwen2-72b-instruct-q6_k-00001-of-00002.gguf",
			expected: true,
		},
		{
			given:    "Grok-100B-v1.0-Q4_0-00003-of-00009.gguf",
			expected: true,
		},
		{
			given:    "Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00001-of-00009.gguf",
			expected: true,
		},
		{
			given:    "Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00001-of-00018.gguf",
			expected: true,
		},
		{
			given:    "not-a-known-arrangement.gguf",
			expected: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.given, func(t *testing.T) {
			actual := IsShardGGUFFilename(tc.given)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestCompleteShardGGUFFilename(t *testing.T) {
	cases := []struct {
		given    string
		expected []string
	}{
		{
			given: "qwen2-72b-instruct-q6_k-00001-of-00002.gguf",
			expected: []string{
				"qwen2-72b-instruct-q6_k-00001-of-00002.gguf",
				"qwen2-72b-instruct-q6_k-00002-of-00002.gguf",
			},
		},
		{
			given: "Grok-100B-v1.0-Q4_0-00003-of-00009.gguf",
			expected: []string{
				"Grok-100B-v1.0-Q4_0-00001-of-00009.gguf",
				"Grok-100B-v1.0-Q4_0-00002-of-00009.gguf",
				"Grok-100B-v1.0-Q4_0-00003-of-00009.gguf",
				"Grok-100B-v1.0-Q4_0-00004-of-00009.gguf",
				"Grok-100B-v1.0-Q4_0-00005-of-00009.gguf",
				"Grok-100B-v1.0-Q4_0-00006-of-00009.gguf",
				"Grok-100B-v1.0-Q4_0-00007-of-00009.gguf",
				"Grok-100B-v1.0-Q4_0-00008-of-00009.gguf",
				"Grok-100B-v1.0-Q4_0-00009-of-00009.gguf",
			},
		},
		{
			given: "Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00001-of-00009.gguf",
			expected: []string{
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00001-of-00009.gguf",
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00002-of-00009.gguf",
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00003-of-00009.gguf",
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00004-of-00009.gguf",
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00005-of-00009.gguf",
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00006-of-00009.gguf",
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00007-of-00009.gguf",
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00008-of-00009.gguf",
				"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00009-of-00009.gguf",
			},
		},
		{
			given: "Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00001-of-00018.gguf",
			expected: []string{
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00001-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00002-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00003-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00004-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00005-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00006-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00007-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00008-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00009-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00010-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00011-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00012-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00013-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00014-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00015-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00016-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00017-of-00018.gguf",
				"Meta-Llama-3.1-405B-Instruct-XelotX-BF16-00018-of-00018.gguf",
			},
		},
		{
			given:    "not-a-known-arrangement.gguf",
			expected: nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.given, func(t *testing.T) {
			actual := CompleteShardGGUFFilename(tc.given)
			assert.Equal(t, tc.expected, actual)
		})
	}
}
