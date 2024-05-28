package httpx

import (
	"net/http"
	"time"
)

type ClientOption struct {
	*TransportOption

	timeout    time.Duration
	debug      bool
	roundTrips []func(req *http.Request) error
}

func ClientOptions() *ClientOption {
	return &ClientOption{
		TransportOption: TransportOptions().WithoutKeepalive(),
		timeout:         30 * time.Second,
	}
}

// WithTransport sets the TransportOption.
func (o *ClientOption) WithTransport(opt *TransportOption) *ClientOption {
	if o == nil || opt == nil {
		return o
	}
	o.TransportOption = opt
	return o
}

// WithTimeout sets the request timeout.
//
// This timeout controls the sum of [network dial], [tls handshake], [request], [response header reading] and [response body reading].
//
// Use 0 to disable timeout.
func (o *ClientOption) WithTimeout(timeout time.Duration) *ClientOption {
	if o == nil || timeout < 0 {
		return o
	}
	o.timeout = timeout
	return o
}

// WithDebug sets the debug mode.
func (o *ClientOption) WithDebug() *ClientOption {
	if o == nil {
		return o
	}
	o.debug = true
	return o
}

// WithRoundTrip sets the round trip function.
func (o *ClientOption) WithRoundTrip(rt func(req *http.Request) error) *ClientOption {
	if o == nil || rt == nil {
		return o
	}
	o.roundTrips = append(o.roundTrips, rt)
	return o
}

// WithUserAgent sets the user agent.
func (o *ClientOption) WithUserAgent(ua string) *ClientOption {
	return o.WithRoundTrip(func(req *http.Request) error {
		req.Header.Set("User-Agent", ua)
		return nil
	})
}

// WithBearerAuth sets the bearer token.
func (o *ClientOption) WithBearerAuth(token string) *ClientOption {
	return o.WithRoundTrip(func(req *http.Request) error {
		req.Header.Set("Authorization", "Bearer "+token)
		return nil
	})
}

// WithBasicAuth sets the basic authentication.
func (o *ClientOption) WithBasicAuth(username, password string) *ClientOption {
	return o.WithRoundTrip(func(req *http.Request) error {
		req.SetBasicAuth(username, password)
		return nil
	})
}

// WithHeader sets the header.
func (o *ClientOption) WithHeader(key, value string) *ClientOption {
	return o.WithRoundTrip(func(req *http.Request) error {
		req.Header.Set(key, value)
		return nil
	})
}

// WithHeaders sets the headers.
func (o *ClientOption) WithHeaders(headers map[string]string) *ClientOption {
	return o.WithRoundTrip(func(req *http.Request) error {
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		return nil
	})
}

// If is a conditional option,
// which receives a boolean condition to trigger the given function or not.
func (o *ClientOption) If(condition bool, then func(*ClientOption) *ClientOption) *ClientOption {
	if condition {
		return then(o)
	}
	return o
}
