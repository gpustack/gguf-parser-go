package httpx

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"syscall"

	"github.com/smallnest/ringbuffer"

	"github.com/gpustack/gguf-parser-go/util/bytex"
)

type SeekerFile struct {
	cli        *http.Client
	req        *http.Request
	b          *ringbuffer.RingBuffer
	c          int64
	l          int64
	objectSize int64
}

// OpenSeekerFile tries the GET http.Request as a SeekerFile,
// and returns a SeekerFile, or an error if any.
func OpenSeekerFile(cli *http.Client, req *http.Request, opts ...*SeekerFileOption) (*SeekerFile, error) {
	if cli == nil {
		return nil, errors.New("client is nil")
	}
	if req == nil {
		return nil, errors.New("request is nil")
	}
	if req.Method != http.MethodGet {
		return nil, errors.New("request method is not GET")
	}

	var o *SeekerFileOption
	if len(opts) > 0 {
		o = opts[0]
	} else {
		o = SeekerFileOptions()
	}
	if o.bufSize <= 0 {
		o.bufSize = 4 * 1024 * 1024 // 4mb
	}

	var l int64
	{
		if !o.skipRangeDownloadDetect {
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
		} else {
			req := req.Clone(req.Context())
			err := Do(cli, req, func(resp *http.Response) error {
				if resp.StatusCode != http.StatusOK {
					return fmt.Errorf("stat: status code %d", resp.StatusCode)
				}
				l = resp.ContentLength
				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("stat: do get request: %w", err)
			}
		}
	}

	if l < 0 {
		return nil, errors.New("remote file size is unknown")
	}
	size := l
	if o.size > 0 {
		if int64(o.size) > l {
			return nil, fmt.Errorf("size %d is greater than limit %d", o.size, l)
		}
		size = int64(o.size)
	}
	// Network requests carry the context. The in-memory ring is nonblocking.
	b := ringbuffer.New(o.bufSize)
	return &SeekerFile{cli: cli, req: req, b: b, l: size, objectSize: l}, nil
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
	if len(p) == 0 {
		return 0, nil
	}
	var n int
	for n < len(p) {
		if off >= f.Len() {
			return n, io.EOF
		}
		if off < f.c || off-f.c >= int64(f.b.Length()) {
			if err := f.sync(off); err != nil {
				return n, err
			}
		}
		if err := f.skip(off - f.c); err != nil {
			return n, err
		}
		read, err := f.Read(p[n:])
		n += read
		off += int64(read)
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func (f *SeekerFile) Read(p []byte) (int, error) {
	n, err := f.b.Read(p)
	f.c += int64(n)
	return n, err
}

func (f *SeekerFile) sync(off int64) error {
	length := min(int64(f.b.Capacity()), f.Len()-off)
	if length <= 0 {
		return io.EOF
	}
	lim := off + length - 1
	req := f.req.Clone(f.req.Context())
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, lim))

	resp, err := f.cli.Do(req)
	if err != nil {
		return err
	}
	defer Close(resp)
	switch resp.StatusCode {
	case http.StatusPartialContent:
		var start, end int64
		span, total, ok := strings.Cut(resp.Header.Get("Content-Range"), "/")
		_, err = fmt.Sscanf(span, "bytes %d-%d", &start, &end)
		if !ok || err != nil || start != off || end < start || end > lim || (total != "*" && total != strconv.FormatInt(f.objectSize, 10)) {
			return errors.New("inconsistent remote byte range")
		}
		length = end - start + 1
	case http.StatusOK:
		if off != 0 || (resp.ContentLength >= 0 && resp.ContentLength != f.objectSize) {
			return errors.New("remote server did not return the requested range")
		}
	default:
		return errors.New(resp.Status)
	}

	buf := bytex.GetBytes()
	defer bytex.Put(buf)
	f.b.Reset()
	f.c = off

	n, err := io.CopyBuffer(_WriterOnly{w: f.b}, io.LimitReader(resp.Body, length), buf)
	if err != nil {
		return err
	}
	if n != length {
		return io.ErrUnexpectedEOF
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

// _WriterOnly is a wrapper to expose the io.Writer method only,
// which to avoid calling the io.ReaderFrom method.
type _WriterOnly struct {
	w io.Writer
}

func (w _WriterOnly) Write(p []byte) (int, error) {
	return w.w.Write(p)
}
