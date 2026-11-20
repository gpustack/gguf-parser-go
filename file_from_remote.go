package gguf_parser

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gpustack/gguf-parser-go/util/httpx"
	"github.com/gpustack/gguf-parser-go/util/osx"
)

// ParseGGUFFileFromHuggingFace parses a GGUF file from Hugging Face(https://huggingface.co/),
// and returns a GGUFFile, or an error if any.
func ParseGGUFFileFromHuggingFace(ctx context.Context, repo, file string, opts ...GGUFReadOption) (*GGUFFile, error) {
	ep := osx.Getenv("HF_ENDPOINT", "https://huggingface.co")
	return ParseGGUFFileRemote(ctx, fmt.Sprintf("%s/%s/resolve/main/%s", ep, repo, file), opts...)
}

// ParseGGUFFileFromModelScope parses a GGUF file from Model Scope(https://modelscope.cn/),
// and returns a GGUFFile, or an error if any.
func ParseGGUFFileFromModelScope(ctx context.Context, repo, file string, opts ...GGUFReadOption) (*GGUFFile, error) {
	ep := osx.Getenv("MS_ENDPOINT", "https://modelscope.cn")
	opts = append(opts[:len(opts):len(opts)], SkipRangeDownloadDetection())
	return ParseGGUFFileRemote(ctx, fmt.Sprintf("%s/models/%s/resolve/master/%s", ep, repo, file), opts...)
}

// ParseGGUFFileRemote parses a GGUF file from a remote BlobURL,
// and returns a GGUFFile, or an error if any.
func ParseGGUFFileRemote(ctx context.Context, url string, opts ...GGUFReadOption) (gf *GGUFFile, err error) {
	var o _GGUFReadOptions
	for _, opt := range opts {
		opt(&o)
	}

	// Cache.
	{
		c := o.parsedFileCache("remote")

		// Get from cache.
		if gf, err = c.Get(url, o.CacheExpiration); err == nil {
			return gf, nil
		}

		// Put to cache.
		defer func() {
			if err == nil {
				_ = c.Put(url, gf)
			}
		}()
	}

	cli := httpx.Client(
		httpx.ClientOptions().
			WithUserAgent("gguf-parser-go").
			If(o.Debug,
				func(x *httpx.ClientOption) *httpx.ClientOption {
					return x.WithDebug()
				},
			).
			If(o.BearerAuthToken != "",
				func(x *httpx.ClientOption) *httpx.ClientOption {
					return x.WithBearerAuth(o.BearerAuthToken)
				},
			).
			If(len(o.Headers) > 0,
				func(x *httpx.ClientOption) *httpx.ClientOption {
					return x.WithHeaders(o.Headers)
				},
			).
			WithTimeout(0).
			WithTransport(
				httpx.TransportOptions().
					WithoutKeepalive().
					TimeoutForDial(5*time.Second).
					TimeoutForTLSHandshake(5*time.Second).
					TimeoutForResponseHeader(5*time.Second).
					If(o.SkipProxy,
						func(x *httpx.TransportOption) *httpx.TransportOption {
							return x.WithoutProxy()
						},
					).
					If(o.ProxyURL != nil,
						func(x *httpx.TransportOption) *httpx.TransportOption {
							return x.WithProxy(http.ProxyURL(o.ProxyURL))
						},
					).
					If(o.SkipTLSVerification || !strings.HasPrefix(url, "https://"),
						func(x *httpx.TransportOption) *httpx.TransportOption {
							return x.WithoutInsecureVerify()
						},
					).
					If(o.SkipDNSCache,
						func(x *httpx.TransportOption) *httpx.TransportOption {
							return x.WithoutDNSCache()
						},
					),
			),
	)

	return parseGGUFFileFromRemote(ctx, cli, url, o)
}

func parseGGUFFileFromRemote(ctx context.Context, cli *http.Client, rawURL string, o _GGUFReadOptions) (*GGUFFile, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}
	files := ggufShardFiles(u.EscapedPath())
	if files.index == 0 {
		files.name = rawURL
	} else {
		// Preserve the authority and query verbatim. Only the filename identifies siblings.
		rawURL = u.String()
		end := strings.IndexAny(rawURL, "?#")
		if end < 0 {
			end = len(rawURL)
		}
		files.prefix = strings.TrimSuffix(rawURL[:end], u.EscapedPath()) + files.prefix
		files.suffix = rawURL[end:]
	}
	if files.count > o.remoteShardLimit() {
		return nil, fmt.Errorf("shard count %d exceeds limit %d", files.count, o.remoteShardLimit())
	}
	ctx, cancel := o.remoteReadContext(ctx)
	defer cancel()

	return parseGGUFFile(files, o, func(path string) (_GGUFFileReadSeeker, error) {
		req, err := httpx.NewGetRequestWithContext(ctx, path)
		if err != nil {
			return _GGUFFileReadSeeker{}, fmt.Errorf("new request: %w", err)
		}
		sf, err := httpx.OpenSeekerFile(cli, req,
			httpx.SeekerFileOptions().
				WithBufferSize(o.BufferSize).
				If(o.SkipRangeDownloadDetection,
					func(x *httpx.SeekerFileOption) *httpx.SeekerFileOption {
						return x.WithoutRangeDownloadDetect()
					},
				),
		)
		if err != nil {
			return _GGUFFileReadSeeker{}, fmt.Errorf("open http file: %w", err)
		}
		return _GGUFFileReadSeeker{sf, io.NewSectionReader(sf, 0, sf.Len()), sf.Len()}, nil
	})
}
