package gguf_parser

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/thxcode/gguf-parser-go/util/ptr"
)

func TestParseGGUFFilename(t *testing.T) {
	cases := []struct {
		given    string
		expected *GGUFFilename
	}{
		{
			given: "Mixtral-v0.1-8x7B-KQ2.gguf",
			expected: &GGUFFilename{
				ModelName:      "Mixtral",
				Major:          ptr.To(0),
				Minor:          ptr.To(1),
				ExpertsCount:   ptr.To(8),
				Parameters:     "7B",
				EncodingScheme: "KQ2",
			},
		},
		{
			given: "Grok-v1.0-100B-Q4_0-00003-of-00009.gguf",
			expected: &GGUFFilename{
				ModelName:      "Grok",
				Major:          ptr.To(1),
				Minor:          ptr.To(0),
				Parameters:     "100B",
				EncodingScheme: "Q4_0",
				Shard:          ptr.To(3),
				ShardTotal:     ptr.To(9),
			},
		},
		{
			given: "Hermes-2-Pro-Llama-3-8B-F16.gguf",
			expected: &GGUFFilename{
				ModelName:      "Hermes 2 Pro Llama 3",
				Parameters:     "8B",
				EncodingScheme: "F16",
			},
		},
		{
			given: "Hermes-2-Pro-Llama-3-v32.33-8Q-F16.gguf",
			expected: &GGUFFilename{
				ModelName:      "Hermes 2 Pro Llama 3",
				Major:          ptr.To(32),
				Minor:          ptr.To(33),
				Parameters:     "8Q",
				EncodingScheme: "F16",
			},
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
				ModelName:      "Mixtral",
				Major:          ptr.To(0),
				Minor:          ptr.To(1),
				ExpertsCount:   ptr.To(8),
				Parameters:     "7B",
				EncodingScheme: "KQ2",
			},
			expected: "Mixtral-v0.1-8x7B-KQ2.gguf",
		},
		{
			given: GGUFFilename{
				ModelName:      "Grok",
				Major:          ptr.To(1),
				Minor:          ptr.To(0),
				Parameters:     "100B",
				EncodingScheme: "Q4_0",
				Shard:          ptr.To(3),
				ShardTotal:     ptr.To(9),
			},
			expected: "Grok-v1.0-100B-Q4_0-00003-of-00009.gguf",
		},
		{
			given: GGUFFilename{
				ModelName:      "Hermes 2 Pro Llama 3",
				Parameters:     "8B",
				EncodingScheme: "F16",
			},
			expected: "Hermes-2-Pro-Llama-3-8B-F16.gguf",
		},
		{
			given: GGUFFilename{
				ModelName:      "Hermes 2 Pro Llama 3",
				Major:          ptr.To(0),
				Minor:          ptr.To(0),
				Parameters:     "8B",
				EncodingScheme: "F16",
			},
			expected: "Hermes-2-Pro-Llama-3-v0.0-8B-F16.gguf",
		},
		{
			given: GGUFFilename{
				ModelName:      "Hermes 2 Pro Llama 3",
				Major:          ptr.To(32),
				Minor:          ptr.To(33),
				Parameters:     "8Q",
				EncodingScheme: "F16",
			},
			expected: "Hermes-2-Pro-Llama-3-v32.33-8Q-F16.gguf",
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
