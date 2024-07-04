package gguf_parser

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"time"

	"golang.org/x/exp/maps"
	"golang.org/x/net/html"

	"github.com/thxcode/gguf-parser-go/util/funcx"
	"github.com/thxcode/gguf-parser-go/util/httpx"
	"github.com/thxcode/gguf-parser-go/util/json"
	"github.com/thxcode/gguf-parser-go/util/stringx"
)

var (
	ErrOllamaInvalidModel      = errors.New("ollama invalid model")
	ErrOllamaBaseLayerNotFound = errors.New("ollama base layer not found")
)

// ParseGGUFFileFromOllama parses a GGUF file from Ollama model's base layer,
// and returns a GGUFFile, or an error if any.
//
// If the crawl is true, it will try to crawl the metadata from Ollama website instead of blobs fetching,
// which will be more efficient and faster, but lossy.
// If the crawling fails, it will fall back to the default behavior.
func ParseGGUFFileFromOllama(ctx context.Context, model string, crawl bool, opts ...GGUFReadOption) (*GGUFFile, error) {
	var o _GGUFReadOptions
	for _, opt := range opts {
		opt(&o)
	}

	om := ParseOllamaModel(model)
	if om == nil {
		return nil, ErrOllamaInvalidModel
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
					}).
					If(o.SkipDNSCache, func(x *httpx.TransportOption) *httpx.TransportOption {
						return x.WithoutDNSCache()
					})))

	var ml OllamaModelLayer
	{
		err := om.Complete(ctx, cli)
		if err != nil {
			return nil, fmt.Errorf("complete ollama model: %w", err)
		}

		var ok bool
		ml, ok = om.GetLayer("application/vnd.ollama.image.model")
		if !ok {
			return nil, ErrOllamaBaseLayerNotFound
		}
	}

	if crawl {
		mwu, lwu := om.WebURL().String(), ml.WebURL().String()
		req, err := httpx.NewGetRequestWithContext(ctx, lwu)
		if err != nil {
			return nil, fmt.Errorf("new request: %w", err)
		}
		req.Header.Add("Referer", mwu)
		req.Header.Add("Hx-Current-Url", mwu)
		req.Header.Add("Hx-Request", "true")
		req.Header.Add("Hx-Target", "file-explorer")

		var n *html.Node
		err = httpx.Do(cli, req, func(resp *http.Response) error {
			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("status code %d", resp.StatusCode)
			}

			n, err = html.Parse(resp.Body)
			if err != nil {
				return fmt.Errorf("parse html: %w", err)
			}

			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("do crawl request: %w", err)
		}

		var wk func(*html.Node) string
		wk = func(n *html.Node) string {
			if n.Type == html.ElementNode && n.Data == "div" {
				for i := range n.Attr {
					if n.Attr[i].Key == "class" && n.Attr[i].Val == "whitespace-pre-wrap" {
						return n.FirstChild.Data
					}
				}
			}
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				if r := wk(c); r != "" {
					return r
				}
			}
			return ""
		}

		r := wk(n)
		if r != "" {
			return parseGGUFFileFromMetadata("ollama", r, ml.Size)
		}

		// Fallback to the default behavior.
	}

	return parseGGUFFileFromRemote(ctx, cli, ml.URL().String(), o)
}

type _OllamaMetadata struct {
	Metadata  map[string]any `json:"metadata"`
	NumParams uint64         `json:"num_params"`
	Tensors   []struct {
		Name   string   `json:"name"`
		Shape  []uint64 `json:"shape"`
		Offset uint64   `json:"offset"`
		Type   uint32   `json:"type"`
	} `json:"tensors"`
	Version uint32 `json:"version"`
}

func parseGGUFFileFromMetadata(source, data string, size uint64) (*GGUFFile, error) {
	if source != "ollama" {
		return nil, fmt.Errorf("invalid source %q", source)
	}

	var m _OllamaMetadata
	if err := json.Unmarshal([]byte(data), &m); err != nil {
		return nil, fmt.Errorf("unmarshal metadata: %w", err)
	}

	arrayMetadataValueRegex := regexp.MustCompile(`^\.{3} \((?P<len>\d+) values\)$`)

	var gf GGUFFile

	gf.Header.Magic = GGUFMagicGGUFLe
	gf.Header.Version = GGUFVersion(m.Version)
	gf.Header.TensorCount = uint64(len(m.Tensors))
	gf.Header.MetadataKVCount = uint64(len(m.Metadata) + 1 /* tokenizer.chat_template */)
	gf.Size = GGUFBytesScalar(size)
	gf.ModelParameters = GGUFParametersScalar(m.NumParams)

	gf.Header.MetadataKV = make([]GGUFMetadataKV, 0, len(m.Metadata))
	for _, k := range func() []string {
		ks := maps.Keys(m.Metadata)
		ks = append(ks, "tokenizer.chat_template")
		sort.Strings(ks)
		return ks
	}() {
		if k == "tokenizer.chat_template" {
			gf.Header.MetadataKV = append(gf.Header.MetadataKV, GGUFMetadataKV{
				Key:       k,
				ValueType: GGUFMetadataValueTypeString,
				Value:     "!!! tokenizer.chat_template !!!",
			})
			continue
		}

		var (
			vt GGUFMetadataValueType
			v  = m.Metadata[k]
		)
		switch vv := v.(type) {
		case bool:
			vt = GGUFMetadataValueTypeBool
		case float64:
			vt = GGUFMetadataValueTypeFloat32
			v = float32(vv)
		case int64:
			vt = GGUFMetadataValueTypeUint32
			v = uint32(vv)
		case string:
			vt = GGUFMetadataValueTypeString
			if r := arrayMetadataValueRegex.FindStringSubmatch(vv); len(r) == 2 {
				vt = GGUFMetadataValueTypeArray
				av := GGUFMetadataKVArrayValue{
					Type: GGUFMetadataValueTypeString,
					Len:  funcx.MustNoError(strconv.ParseUint(r[1], 10, 64)),
				}
				switch _, d, _ := stringx.CutFromRight(k, "."); d {
				case "scores":
					av.Type = GGUFMetadataValueTypeFloat32
				case "token_type":
					av.Type = GGUFMetadataValueTypeInt32
				}
				v = av
			}
		}
		gf.Header.MetadataKV = append(gf.Header.MetadataKV, GGUFMetadataKV{
			Key:       k,
			ValueType: vt,
			Value:     v,
		})
	}

	gf.TensorInfos = make([]GGUFTensorInfo, 0, len(m.Tensors))
	for i := range m.Tensors {
		t := m.Tensors[i]
		ti := GGUFTensorInfo{
			Name:        t.Name,
			NDimensions: uint32(len(t.Shape)),
			Dimensions:  t.Shape,
			Offset:      t.Offset,
			Type:        GGMLType(t.Type),
		}
		gf.TensorInfos = append(gf.TensorInfos, ti)
		gf.ModelSize += GGUFBytesScalar(ti.Bytes())
	}

	gf.TensorDataStartOffset = int64(gf.Size - gf.ModelSize)

	if gf.ModelParameters != 0 {
		gf.ModelBitsPerWeight = GGUFBitsPerWeightScalar(float64(gf.ModelSize) * 8 / float64(gf.ModelParameters))
	}

	return &gf, nil
}
