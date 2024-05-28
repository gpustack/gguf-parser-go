package httpx

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"syscall"

	"github.com/smallnest/ringbuffer"

	"github.com/thxcode/gguf-parser-go/util/bytex"
)

type SeekerFile struct {
	cli *http.Client
	req *http.Request
	b   *ringbuffer.RingBuffer
	c   int64
	l   int64
}

func OpenSeekerFile(cli *http.Client, req *http.Request) (*SeekerFile, error) {
	return OpenSeekerFileWithSize(cli, req, 0, 0)
}

func OpenSeekerFileWithSize(cli *http.Client, req *http.Request, bufSize, size int) (*SeekerFile, error) {
	if cli == nil {
		return nil, errors.New("client is nil")
	}
	if req == nil {
		return nil, errors.New("request is nil")
	}
	if req.Method != http.MethodGet {
		return nil, errors.New("request method is not GET")
	}

	var l int64
	{
		req := req.Clone(req.Context())
		req.Method = http.MethodHead
		err := Do(cli, req, func(resp *http.Response) error {
			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("stat: status code %d", resp.StatusCode)
			}
			if !strings.EqualFold(resp.Header.Get("Accept-Ranges"), "bytes") {
				return fmt.Errorf("stat: not support range download")
			}
			l = resp.ContentLength
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("stat: do head request: %w", err)
		}
		switch sz := int64(size); {
		case sz > l:
			return nil, fmt.Errorf("size %d is greater than limit %d", size, l)
		case sz <= 0:
		default:
			l = sz
		}
	}

	if bufSize <= 0 {
		bufSize = 4 * 1024 * 1024 // 4mb
	}

	b := ringbuffer.New(bufSize).WithCancel(req.Context())
	return &SeekerFile{cli: cli, req: req, b: b, c: 1<<63 - 1, l: l}, nil
}

func (f *SeekerFile) Close() error {
	if f.b != nil {
		f.b.CloseWriter()
	}
	return nil
}

func (f *SeekerFile) Len() int64 {
	return f.l
}

func (f *SeekerFile) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, syscall.EINVAL
	}
	if off > f.Len() {
		return 0, io.EOF
	}

	// Sync and move to new offset, if backward or empty buffer.
	if f.c > off || f.b.IsEmpty() {
		if err := f.sync(off, true); err != nil {
			return 0, err
		}
	}

	var (
		remain   = int64(f.b.Length())
		capacity = int64(f.b.Capacity())
		need     = int64(len(p))
	)

	switch {
	case f.c+remain >= off+need: // Skip and move to new offset, if enough to forward.
		if err := f.skip(off - f.c); err != nil {
			return 0, err
		}
		return f.Read(p)
	case f.c+capacity >= off+need: // Sync and move to new offset, if enough to forward after synced.
		if err := f.sync(f.c+remain, false); err != nil {
			return 0, err
		}
		if err := f.skip(off - f.c); err != nil {
			return 0, err
		}
		return f.Read(p)
	default:
	}

	// Otherwise, read directly.

	f.b.Reset()
	f.c = off

	// Request remain needing.
	lim := off + int64(len(p)) - 1
	if lim > f.Len() {
		lim = f.Len()
	}
	req := f.req.Clone(f.req.Context())
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, lim))
	resp, err := f.cli.Do(req)
	if err != nil {
		return 0, err
	}
	defer Close(resp)
	if resp.StatusCode != http.StatusPartialContent && resp.StatusCode != http.StatusOK {
		return 0, errors.New(resp.Status)
	}
	n, err := resp.Body.Read(p)
	f.c += int64(n)
	return n, err
}

func (f *SeekerFile) Read(p []byte) (int, error) {
	n, err := f.b.Read(p)
	f.c += int64(n)
	return n, err
}

func (f *SeekerFile) sync(off int64, reset bool) error {
	lim := off + int64(f.b.Free()) - 1
	if lim > f.Len() {
		lim = f.Len()
	}
	req := f.req.Clone(f.req.Context())
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, lim))

	resp, err := f.cli.Do(req)
	if err != nil {
		return err
	}
	defer Close(resp)
	if resp.StatusCode != http.StatusPartialContent && resp.StatusCode != http.StatusOK {
		return errors.New(resp.Status)
	}

	buf := bytex.GetBytes()
	defer bytex.Put(buf)
	if reset {
		f.b.Reset()
		f.c = off
	}
	_, err = io.CopyBuffer(f.b, resp.Body, buf)
	if err != nil {
		return err
	}

	return nil
}

func (f *SeekerFile) skip(dif int64) error {
	if dif <= 0 {
		return nil
	}

	buf := bytex.GetBytes(uint64(dif))
	defer bytex.Put(buf)
	n, err := f.b.Read(buf)
	f.c += int64(n)
	if err != nil {
		return err
	}
	return nil
}
