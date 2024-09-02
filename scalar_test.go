package gguf_parser

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseSizeScalar(t *testing.T) {
	testCases := []struct {
		given    string
		expected SizeScalar
	}{
		{"1", 1},
		{"1K", 1 * _Ki},
		{"1M", 1 * _Mi},
		{"1G", 1 * _Gi},
		{"1T", 1 * _Ti},
		{"1P", 1 * _Pi},
	}
	for _, tc := range testCases {
		t.Run(tc.given, func(t *testing.T) {
			actual, err := ParseSizeScalar(tc.given)
			if !assert.NoError(t, err) {
				return
			}
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestParseFLOPSScalar(t *testing.T) {
	testCases := []struct {
		given    string
		expected FLOPSScalar
	}{
		{"1FLOPS", 1},
		{"1KFLOPS", 1 * _K},
		{"1MFLOPS", 1 * _M},
		{"1GFLOPS", 1 * _G},
		{"1TFLOPS", 1 * _T},
		{"1PFLOPS", 1 * _P},
	}
	for _, tc := range testCases {
		t.Run(tc.given, func(t *testing.T) {
			actual, err := ParseFLOPSScalar(tc.given)
			if !assert.NoError(t, err) {
				return
			}
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestParseBytesPerSecondScalar(t *testing.T) {
	testCases := []struct {
		given    string
		expected BytesPerSecondScalar
	}{
		{"1B/s", 1},
		{"1KB/s", 1 * _K},
		{"1MB/s", 1 * _M},
		{"1GB/s", 1 * _G},
		{"1TB/s", 1 * _T},
		{"1PB/s", 1 * _P},
		{"1KiBps", 1 * _Ki},
		{"1MiBps", 1 * _Mi},
		{"1GiBps", 1 * _Gi},
		{"1TiBps", 1 * _Ti},
		{"1PiBps", 1 * _Pi},
		{"8b/s", 1},
		{"1Kbps", 1 * _K >> 3},
		{"1Mbps", 1 * _M >> 3},
		{"1Gbps", 1 * _G >> 3},
		{"1Tbps", 1 * _T >> 3},
		{"1Pbps", 1 * _P >> 3},
		{"1Kibps", 1 * _Ki >> 3},
		{"1Mibps", 1 * _Mi >> 3},
		{"1Gibps", 1 * _Gi >> 3},
		{"1Tibps", 1 * _Ti >> 3},
		{"1Pibps", 1 * _Pi >> 3},
	}
	for _, tc := range testCases {
		t.Run(tc.given, func(t *testing.T) {
			actual, err := ParseBytesPerSecondScalar(tc.given)
			if !assert.NoError(t, err) {
				return
			}
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestParseGGUFBytesScalar(t *testing.T) {
	testCases := []struct {
		given    string
		expected GGUFBytesScalar
	}{
		{"1B", 1},
		{"1KB", 1 * _K},
		{"1MB", 1 * _M},
		{"1GB", 1 * _G},
		{"1TB", 1 * _T},
		{"1PB", 1 * _P},
		{"1KiB", 1 * _Ki},
		{"1MiB", 1 * _Mi},
		{"1GiB", 1 * _Gi},
		{"1TiB", 1 * _Ti},
		{"1PiB", 1 * _Pi},
	}
	for _, tc := range testCases {
		t.Run(tc.given, func(t *testing.T) {
			actual, err := ParseGGUFBytesScalar(tc.given)
			if !assert.NoError(t, err) {
				return
			}
			assert.Equal(t, tc.expected, actual)
		})
	}
}
