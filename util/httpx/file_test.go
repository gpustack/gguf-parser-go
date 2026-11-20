package httpx

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSeekerFileRanges(t *testing.T) {
	content := []byte("0123456789abcdef")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "test", time.Time{}, bytes.NewReader(content))
	}))
	defer srv.Close()
	req, err := NewGetRequestWithContext(context.Background(), srv.URL)
	require.NoError(t, err)
	f, err := OpenSeekerFile(srv.Client(), req, SeekerFileOptions().WithBufferSize(4))
	require.NoError(t, err)
	defer f.Close()
	for _, tc := range []struct {
		offset int64
		size   int
		want   string
		err    error
	}{
		{2, 7, "2345678", nil},
		{0, 3, "012", nil},
		{3, 1, "3", nil},
		{14, 4, "ef", io.EOF},
	} {
		b := make([]byte, tc.size)
		n, err := f.ReadAt(b, tc.offset)
		require.ErrorIs(t, err, tc.err)
		require.Equal(t, tc.want, string(b[:n]))
	}
}

func TestSeekerFileChecksRangeSize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Accept-Ranges", "bytes")
			w.Header().Set("Content-Length", "12")
			return
		}
		w.Header().Set("Content-Range", "bytes 0-3/10")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte("0123"))
	}))
	defer srv.Close()
	req, err := NewGetRequestWithContext(context.Background(), srv.URL)
	require.NoError(t, err)
	f, err := OpenSeekerFile(srv.Client(), req, SeekerFileOptions().WithBufferSize(4))
	require.NoError(t, err)
	defer f.Close()
	var b [1]byte
	_, err = f.ReadAt(b[:], 0)
	require.ErrorContains(t, err, "inconsistent remote byte range")
}
