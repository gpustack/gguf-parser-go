package gguf_parser

import (
	"context"
	"net/url"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/gpustack/gguf-parser-go/util/osx"
)

// DefaultMetadataReadBudget is the metadata accounting allowance per model, in bytes.
const DefaultMetadataReadBudget = 256 << 20

// DefaultRemoteShardLimit is the maximum number of remote shards per model by default.
const DefaultRemoteShardLimit = 2048

// DefaultRemoteReadTimeout bounds a remote model read when the caller supplies no deadline.
const DefaultRemoteReadTimeout = 5 * time.Minute

type (
	_GGUFReadOptions struct {
		Debug              bool
		SkipLargeMetadata  bool
		MetadataReadBudget uint64
		RemoteShardLimit   int
		RemoteReadTimeout  time.Duration

		// Local.
		MMap bool

		// Remote.
		BearerAuthToken            string
		Headers                    map[string]string
		ProxyURL                   *url.URL
		SkipProxy                  bool
		SkipTLSVerification        bool
		SkipDNSCache               bool
		BufferSize                 int
		SkipRangeDownloadDetection bool
		CachePath                  string
		CacheExpiration            time.Duration
	}

	// GGUFReadOption is the option for reading the file.
	GGUFReadOption func(o *_GGUFReadOptions)
)

// UseDebug uses debug mode to read the file.
func UseDebug() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.Debug = true
	}
}

// SkipLargeMetadata skips reading large GGUFMetadataKV items,
// which are not necessary for most cases.
func SkipLargeMetadata() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.SkipLargeMetadata = true
	}
}

// UseMetadataReadBudget sets the accounting budget in bytes for a complete GGUF read.
// The default is 256 MiB. Zero selects the default.
func UseMetadataReadBudget(bytes uint64) GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.MetadataReadBudget = bytes
	}
}

func (o _GGUFReadOptions) metadataBudget() uint64 {
	if o.MetadataReadBudget == 0 {
		return DefaultMetadataReadBudget
	}
	return o.MetadataReadBudget
}

// UseRemoteShardLimit sets the maximum number of files to open for a remote GGUF model.
// The default is 2,048. Values below one select the default. GGUF filenames allow up to 99,999 shards.
func UseRemoteShardLimit(count int) GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.RemoteShardLimit = count
	}
}

func (o _GGUFReadOptions) remoteShardLimit() int {
	if o.RemoteShardLimit < 1 {
		return DefaultRemoteShardLimit
	}
	return min(o.RemoteShardLimit, 99999)
}

// UseRemoteReadTimeout sets a time limit for reading one remote model, including all its shards.
// A shorter caller deadline takes precedence. Nonpositive durations select the default behavior.
func UseRemoteReadTimeout(timeout time.Duration) GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.RemoteReadTimeout = timeout
	}
}

func (o _GGUFReadOptions) remoteReadContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if o.RemoteReadTimeout > 0 {
		return context.WithTimeout(ctx, o.RemoteReadTimeout)
	}
	if _, ok := ctx.Deadline(); ok {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, DefaultRemoteReadTimeout)
}

// UseMMap uses mmap to read the local file.
func UseMMap() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.MMap = true
	}
}

// UseBearerAuth uses the given token as a bearer auth when reading from remote.
func UseBearerAuth(token string) GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.BearerAuthToken = token
	}
}

// UseHeaders uses the given headers when reading from remote.
func UseHeaders(headers map[string]string) GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.Headers = headers
	}
}

// UseProxy uses the given url as a proxy when reading from remote.
func UseProxy(url *url.URL) GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.ProxyURL = url
	}
}

// SkipProxy skips the proxy when reading from remote.
func SkipProxy() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.SkipProxy = true
	}
}

// SkipTLSVerification skips the TLS verification when reading from remote.
func SkipTLSVerification() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.SkipTLSVerification = true
	}
}

// SkipDNSCache skips the DNS cache when reading from remote.
func SkipDNSCache() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.SkipDNSCache = true
	}
}

// UseBufferSize sets the buffer size when reading from remote.
func UseBufferSize(size int) GGUFReadOption {
	const minSize = 32 * 1024
	if size < minSize {
		size = minSize
	}
	return func(o *_GGUFReadOptions) {
		o.BufferSize = size
	}
}

// SkipRangeDownloadDetection skips the range download detection when reading from remote.
func SkipRangeDownloadDetection() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.SkipRangeDownloadDetection = true
	}
}

// UseCache caches the remote reading result.
func UseCache() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.CachePath = DefaultCachePath()
		o.CacheExpiration = 24 * time.Hour
	}
}

// SkipCache skips the cache when reading from remote.
func SkipCache() GGUFReadOption {
	return func(o *_GGUFReadOptions) {
		o.CachePath = ""
		o.CacheExpiration = 0
	}
}

// DefaultCachePath returns the default cache path.
func DefaultCachePath() string {
	cd := filepath.Join(osx.UserHomeDir(), ".cache")
	if runtime.GOOS == "windows" {
		cd = osx.Getenv("APPDATA", cd)
	}
	return filepath.Join(cd, "gguf-parser")
}

// UseCachePath uses the given path to cache the remote reading result.
func UseCachePath(path string) GGUFReadOption {
	path = strings.TrimSpace(filepath.Clean(osx.InlineTilde(path)))
	return func(o *_GGUFReadOptions) {
		if path == "" {
			return
		}
		o.CachePath = path
	}
}

// UseCacheExpiration uses the given expiration to cache the remote reading result.
//
// Disable cache expiration by setting it to 0.
func UseCacheExpiration(expiration time.Duration) GGUFReadOption {
	if expiration < 0 {
		expiration = 0
	}
	return func(o *_GGUFReadOptions) {
		o.CacheExpiration = expiration
	}
}
