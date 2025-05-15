package httpx

import (
	"context"
	"net"
	"slices"
	"strings"
	"time"

	"github.com/rs/dnscache"
)

// DefaultResolver is the default DNS resolver used by the package,
// which caches DNS lookups in memory.
var DefaultResolver = &dnscache.Resolver{
	// NB(thxCode): usually, a high latency DNS is about 3s,
	// so we set the timeout to 5s here.
	Timeout:  5 * time.Second,
	Resolver: net.DefaultResolver,
}

func init() {
	go func() {
		t := time.NewTimer(5 * time.Minute)
		defer t.Stop()
		for range t.C {
			DefaultResolver.RefreshWithOptions(dnscache.ResolverRefreshOptions{
				ClearUnused:      true,
				PersistOnFailure: false,
			})
		}
	}()
}

func DNSCacheDialContext(dialer *net.Dialer) func(context.Context, string, string) (net.Conn, error) {
	return func(ctx context.Context, nw, addr string) (conn net.Conn, err error) {
		h, p, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, err
		}
		ips, err := DefaultResolver.LookupHost(ctx, h)
		if err != nil {
			return nil, err
		}
		switch len(ips) {
		case 0:
			return nil, net.UnknownNetworkError("failed to resolve host")
		case 1:
			return dialer.DialContext(ctx, nw, net.JoinHostPort(ips[0], p))
		default:
		}
		// Sort IPs to put IPv4 first, then IPv6.
		slices.SortFunc(ips, func(a, b string) int {
			aIPv4, bIPv4 := strings.Contains(a, "."), strings.Contains(b, ".")
			if (aIPv4 && bIPv4) || (!aIPv4 && !bIPv4) {
				return 0
			}
			if !aIPv4 {
				return 1
			}
			return -1
		})
		// Try to connect to each IP address in order.
		for _, ip := range ips {
			conn, err = dialer.DialContext(ctx, nw, net.JoinHostPort(ip, p))
			if err == nil {
				break
			}
		}
		return conn, err
	}
}
