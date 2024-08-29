package gguf_parser

import (
	"errors"
	"strconv"
	"strings"
)

const (
	_Ki = 1 << ((iota + 1) * 10)
	_Mi
	_Gi
	_Ti
	_Pi
)

const (
	_K = 1e3
	_M = 1e6
	_G = 1e9
	_T = 1e12
	_P = 1e15
)

const (
	_Thousand    = 1e3
	_Million     = 1e6
	_Billion     = 1e9
	_Trillion    = 1e12
	_Quadrillion = 1e15
)

type (
	// SizeScalar is the scalar for size.
	SizeScalar uint64

	// FLOPSScalar is the scalar for FLOPS.
	FLOPSScalar uint64

	// BytesPerSecondScalar is the scalar for bytes per second (Bps).
	BytesPerSecondScalar uint64
)

var (
	// _SizeBaseUnitMatrix is the base unit matrix for size.
	_SizeBaseUnitMatrix = []struct {
		Base float64
		Unit string
	}{
		{_Pi, "P"},
		{_Ti, "T"},
		{_Gi, "G"},
		{_Mi, "M"},
		{_Ki, "K"},
	}

	// _BytesBaseUnitMatrix is the base unit matrix for bytes.
	_BytesBaseUnitMatrix = []struct {
		Base float64
		Unit string
	}{
		{_Pi, "Pi"},
		{_P, "P"},
		{_Ti, "Ti"},
		{_T, "T"},
		{_Gi, "Gi"},
		{_G, "G"},
		{_Mi, "Mi"},
		{_M, "M"},
		{_Ki, "Ki"},
		{_K, "K"},
	}

	// _NumberBaseUnitMatrix is the base unit matrix for numbers.
	_NumberBaseUnitMatrix = []struct {
		Base float64
		Unit string
	}{
		{_Quadrillion, "Q"},
		{_Trillion, "T"},
		{_Billion, "B"},
		{_Million, "M"},
		{_Thousand, "K"},
	}
)

// ParseSizeScalar parses the SizeScalar from the string.
func ParseSizeScalar(s string) (_ SizeScalar, err error) {
	if s == "" {
		return 0, errors.New("invalid SizeScalar")
	}
	b := float64(0)
	for i := range _SizeBaseUnitMatrix {
		if strings.HasSuffix(s, _SizeBaseUnitMatrix[i].Unit) {
			b = _SizeBaseUnitMatrix[i].Base
			s = strings.TrimSpace(strings.TrimSuffix(s, _SizeBaseUnitMatrix[i].Unit))
			break
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	return SizeScalar(f * b), nil
}

func (s SizeScalar) String() string {
	if s == 0 {
		return "0"
	}
	b, u := float64(1), ""
	for i := range _SizeBaseUnitMatrix {
		if float64(s) >= _SizeBaseUnitMatrix[i].Base {
			b = _SizeBaseUnitMatrix[i].Base
			u = _SizeBaseUnitMatrix[i].Unit
			break
		}
	}
	f := strconv.FormatFloat(float64(s)/b, 'f', 2, 64)
	return strings.TrimSuffix(f, ".00") + " " + u
}

// ParseFLOPSScalar parses the FLOPSScalar from the string.
func ParseFLOPSScalar(s string) (_ FLOPSScalar, err error) {
	if s == "" {
		return 0, errors.New("invalid FLOPSScalar")
	}
	s = strings.TrimSpace(strings.TrimSuffix(s, "FLOPS"))
	b := float64(0)
	for i := range _SizeBaseUnitMatrix {
		if strings.HasSuffix(s, _SizeBaseUnitMatrix[i].Unit) {
			b = _SizeBaseUnitMatrix[i].Base
			s = strings.TrimSpace(strings.TrimSuffix(s, _SizeBaseUnitMatrix[i].Unit))
			break
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	return FLOPSScalar(f * b), nil
}

func (s FLOPSScalar) String() string {
	if s == 0 {
		return "0 FLOPS"
	}
	b, u := float64(1), ""
	for i := range _SizeBaseUnitMatrix {
		if float64(s) >= _SizeBaseUnitMatrix[i].Base {
			b = _SizeBaseUnitMatrix[i].Base
			u = _SizeBaseUnitMatrix[i].Unit
			break
		}
	}
	f := strconv.FormatFloat(float64(s)/b, 'f', 2, 64)
	return strings.TrimSuffix(f, ".00") + " " + u + "FLOPS"
}

// ParseBytesPerSecondScalar parses the BytesPerSecondScalar from the string.
func ParseBytesPerSecondScalar(s string) (_ BytesPerSecondScalar, err error) {
	if s == "" {
		return 0, errors.New("invalid BytesPerSecondScalar")
	}
	s = strings.TrimSpace(strings.TrimSuffix(s, "Bps"))
	b := float64(0)
	for i := range _BytesBaseUnitMatrix {
		if strings.HasSuffix(s, _BytesBaseUnitMatrix[i].Unit) {
			b = _BytesBaseUnitMatrix[i].Base
			s = strings.TrimSpace(strings.TrimSuffix(s, _BytesBaseUnitMatrix[i].Unit))
			break
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	return BytesPerSecondScalar(f * b), nil
}

func (s BytesPerSecondScalar) String() string {
	if s == 0 {
		return "0 Bps"
	}
	b, u := float64(1), ""
	for i := range _BytesBaseUnitMatrix {
		if float64(s) >= _BytesBaseUnitMatrix[i].Base {
			b = _BytesBaseUnitMatrix[i].Base
			u = _BytesBaseUnitMatrix[i].Unit
			break
		}
	}
	f := strconv.FormatFloat(float64(s)/b, 'f', 2, 64)
	return strings.TrimSuffix(f, ".00") + " " + u + "Bps"
}

type (
	// GGUFBytesScalar is the scalar for bytes.
	GGUFBytesScalar uint64

	// GGUFParametersScalar is the scalar for parameters.
	GGUFParametersScalar uint64

	// GGUFBitsPerWeightScalar is the scalar for bits per weight.
	GGUFBitsPerWeightScalar float64

	// GGUFTokensPerSecondScalar is the scalar for tokens per second.
	GGUFTokensPerSecondScalar float64
)

// ParseGGUFBytesScalar parses the GGUFBytesScalar from the string.
func ParseGGUFBytesScalar(s string) (_ GGUFBytesScalar, err error) {
	if s == "" {
		return 0, errors.New("invalid GGUFBytesScalar")
	}
	s = strings.TrimSpace(strings.TrimSuffix(s, "B"))
	b := float64(0)
	for i := range _BytesBaseUnitMatrix {
		if strings.HasSuffix(s, _BytesBaseUnitMatrix[i].Unit) {
			b = _BytesBaseUnitMatrix[i].Base
			s = strings.TrimSpace(strings.TrimSuffix(s, _BytesBaseUnitMatrix[i].Unit))
			break
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	return GGUFBytesScalar(f * b), nil
}

// GGUFBytesScalarStringInMiBytes is the flag to show the GGUFBytesScalar string in MiB.
var GGUFBytesScalarStringInMiBytes bool

func (s GGUFBytesScalar) String() string {
	if s == 0 {
		return "0 B"
	}
	b, u := float64(1), ""
	if GGUFBytesScalarStringInMiBytes {
		b = _Mi
		u = "Mi"
	} else {
		for i := range _BytesBaseUnitMatrix {
			if float64(s) >= _BytesBaseUnitMatrix[i].Base {
				b = _BytesBaseUnitMatrix[i].Base
				u = _BytesBaseUnitMatrix[i].Unit
				break
			}
		}
	}
	f := strconv.FormatFloat(float64(s)/b, 'f', 2, 64)
	return strings.TrimSuffix(f, ".00") + " " + u + "B"
}

func (s GGUFParametersScalar) String() string {
	if s == 0 {
		return "0"
	}
	b, u := float64(1), ""
	for i := range _NumberBaseUnitMatrix {
		if float64(s) >= _NumberBaseUnitMatrix[i].Base {
			b = _NumberBaseUnitMatrix[i].Base
			u = _NumberBaseUnitMatrix[i].Unit
			break
		}
	}
	f := strconv.FormatFloat(float64(s)/b, 'f', 2, 64)
	return strings.TrimSuffix(f, ".00") + " " + u
}

func (s GGUFBitsPerWeightScalar) String() string {
	if s <= 0 {
		return "0 bpw"
	}
	return strconv.FormatFloat(float64(s), 'f', 2, 64) + " bpw"
}

func (s GGUFTokensPerSecondScalar) String() string {
	if s <= 0 {
		return "0 tps"
	}
	return strconv.FormatFloat(float64(s), 'f', 2, 64) + " tps"
}
