package gguf_parser

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseOllamaModel(t *testing.T) {
	cases := []struct {
		given    string
		expected *OllamaModel
	}{
		{
			given: "gemma2",
			expected: &OllamaModel{
				Schema:     OllamaDefaultScheme,
				Registry:   OllamaDefaultRegistry,
				Namespace:  OllamaDefaultNamespace,
				Repository: "gemma2",
				Tag:        OllamaDefaultTag,
			},
		},
		{
			given: "gemma2:awesome",
			expected: &OllamaModel{
				Schema:     OllamaDefaultScheme,
				Registry:   OllamaDefaultRegistry,
				Namespace:  OllamaDefaultNamespace,
				Repository: "gemma2",
				Tag:        "awesome",
			},
		},
		{
			given: "gemma2:awesome@sha256:1234567890abcdef",
			expected: &OllamaModel{
				Schema:     OllamaDefaultScheme,
				Registry:   OllamaDefaultRegistry,
				Namespace:  OllamaDefaultNamespace,
				Repository: "gemma2",
				Tag:        "awesome",
			},
		},
		{
			given: "awesome/gemma2:latest@sha256:1234567890abcdef",
			expected: &OllamaModel{
				Schema:     OllamaDefaultScheme,
				Registry:   OllamaDefaultRegistry,
				Namespace:  "awesome",
				Repository: "gemma2",
				Tag:        "latest",
			},
		},
		{
			given: "mysite.com/library/gemma2:latest@sha256:1234567890abcdef",
			expected: &OllamaModel{
				Schema:     OllamaDefaultScheme,
				Registry:   "mysite.com",
				Namespace:  "library",
				Repository: "gemma2",
				Tag:        "latest",
			},
		},
		{
			given: "http://mysite.com/library/gemma2:latest@sha256:1234567890abcdef",
			expected: &OllamaModel{
				Schema:     "http",
				Registry:   "mysite.com",
				Namespace:  "library",
				Repository: "gemma2",
				Tag:        "latest",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.given, func(t *testing.T) {
			actual := ParseOllamaModel(tc.given)
			assert.Equal(t, tc.expected, actual)
		})
	}
}
