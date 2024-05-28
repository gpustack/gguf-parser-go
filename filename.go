package gguf_parser

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/thxcode/gguf-parser-go/util/funcx"
	"github.com/thxcode/gguf-parser-go/util/ptr"
)

// GGUFFilename represents a GGUF filename,
// see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#gguf-naming-convention.
type GGUFFilename struct {
	ModelName      string `json:"modelName"`
	Major          *int   `json:"major"`
	Minor          *int   `json:"minor"`
	ExpertsCount   *int   `json:"expertsCount,omitempty"`
	Parameters     string `json:"parameters"`
	EncodingScheme string `json:"encodingScheme"`
	Shard          *int   `json:"shard,omitempty"`
	ShardTotal     *int   `json:"shardTotal,omitempty"`
}

var GGUFFilenameRegex = regexp.MustCompile(`^(?P<model_name>[A-Za-z0-9\s-]+)(?:-v(?P<major>\d+)\.(?P<minor>\d+))?-(?:(?P<experts_count>\d+)x)?(?P<parameters>\d+[A-Za-z]?)-(?P<encoding_scheme>[\w_]+)(?:-(?P<shard>\d{5})-of-(?P<shardTotal>\d{5}))?\.gguf$`) // nolint:lll

// ParseGGUFFilename parses the given GGUF filename string,
// and returns the GGUFFilename, or nil if the filename is invalid.
func ParseGGUFFilename(name string) *GGUFFilename {
	parseInt := func(v string) int {
		return int(funcx.MustNoError(strconv.ParseInt(v, 10, 64)))
	}

	n := name
	if !strings.HasSuffix(n, ".gguf") {
		n += ".gguf"
	}

	m := make(map[string]string)
	{
		r := GGUFFilenameRegex.FindStringSubmatch(n)
		for i, ne := range GGUFFilenameRegex.SubexpNames() {
			if i != 0 && i <= len(r) {
				m[ne] = r[i]
			}
		}
	}

	if m["model_name"] == "" || m["parameters"] == "" || m["encoding_scheme"] == "" {
		return nil
	}

	var gn GGUFFilename

	gn.ModelName = strings.ReplaceAll(m["model_name"], "-", " ")
	if v := m["major"]; v != "" {
		gn.Major = ptr.To(parseInt(v))
	}
	if v := m["minor"]; v != "" {
		gn.Minor = ptr.To(parseInt(v))
	}
	if v := m["experts_count"]; v != "" {
		gn.ExpertsCount = ptr.To(parseInt(v))
	}
	gn.Parameters = m["parameters"]
	gn.EncodingScheme = m["encoding_scheme"]
	if v := m["shard"]; v != "" {
		gn.Shard = ptr.To(parseInt(v))
	}
	if v := m["shardTotal"]; v != "" {
		gn.ShardTotal = ptr.To(parseInt(v))
	}
	return &gn
}

func (gn GGUFFilename) String() string {
	if gn.ModelName == "" || gn.Parameters == "" || gn.EncodingScheme == "" {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(strings.ReplaceAll(gn.ModelName, " ", "-"))
	sb.WriteString("-")
	if gn.Major != nil {
		sb.WriteString("v")
		sb.WriteString(strconv.Itoa(ptr.Deref(gn.Major, 0)))
		sb.WriteString(".")
		sb.WriteString(strconv.Itoa(ptr.Deref(gn.Minor, 0)))
		sb.WriteString("-")
	}
	if v := ptr.Deref(gn.ExpertsCount, 0); v > 0 {
		sb.WriteString(strconv.Itoa(v))
		sb.WriteString("x")
	}
	sb.WriteString(gn.Parameters)
	sb.WriteString("-")
	sb.WriteString(gn.EncodingScheme)
	if m, n := ptr.Deref(gn.Shard, 0), ptr.Deref(gn.ShardTotal, 0); m > 0 && n > 0 {
		sb.WriteString("-")
		sb.WriteString(fmt.Sprintf("%05d", m))
		sb.WriteString("-of-")
		sb.WriteString(fmt.Sprintf("%05d", n))
	}
	sb.WriteString(".gguf")
	return sb.String()
}

func (gn GGUFFilename) IsPreRelease() bool {
	return ptr.Deref(gn.Major, 0) == 0 && ptr.Deref(gn.Minor, 0) == 0
}

func (gn GGUFFilename) IsSharding() bool {
	return ptr.Deref(gn.Shard, 0) > 0 && ptr.Deref(gn.ShardTotal, 0) > 0
}
