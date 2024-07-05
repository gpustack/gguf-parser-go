package gguf_parser

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/thxcode/gguf-parser-go/util/httpx"
	"github.com/thxcode/gguf-parser-go/util/osx"
)

// ParseGGUFFileFromHuggingFace parses a GGUF file from Hugging Face,
// and returns a GGUFFile, or an error if any.
func ParseGGUFFileFromHuggingFace(ctx context.Context, repo, file string, opts ...GGUFReadOption) (*GGUFFile, error) {
	return ParseGGUFFileRemote(ctx, fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, file), opts...)
}

// ParseGGUFFileRemote parses a GGUF file from a remote BlobURL,
// and returns a GGUFFile, or an error if any.
func ParseGGUFFileRemote(ctx context.Context, url string, opts ...GGUFReadOption) (*GGUFFile, error) {
	var o _GGUFReadOptions
	for _, opt := range opts {
		opt(&o)
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

	return parseGGUFFileFromRemote(ctx, cli, url, o)
}

func parseGGUFFileFromRemote(ctx context.Context, cli *http.Client, url string, o _GGUFReadOptions) (*GGUFFile, error) {
	var (
		f io.ReadSeeker
		s int64
	)
	{
		req, err := httpx.NewGetRequestWithContext(ctx, url)
		if err != nil {
			return nil, fmt.Errorf("new request: %w", err)
		}

		var sf *httpx.SeekerFile
		if o.BufferSize > 0 {
			sf, err = httpx.OpenSeekerFileWithSize(cli, req, o.BufferSize, 0)
		} else {
			sf, err = httpx.OpenSeekerFile(cli, req)
		}
		if err != nil {
			return nil, fmt.Errorf("open http file: %w", err)
		}
		defer osx.Close(sf)
		f = io.NewSectionReader(sf, 0, sf.Len())
		s = sf.Len()
	}

	return parseGGUFFile(s, f, o)
}
