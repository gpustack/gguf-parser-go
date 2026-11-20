package gguf_parser

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/gpustack/gguf-parser-go/util/httpx"
)

func TestLayerCountRange(t *testing.T) {
	for _, key := range []string{"llama.block_count", "clip.vision.block_count", "clip.audio.block_count"} {
		t.Run(key, func(t *testing.T) {
			var model []GGUFMetadataKV
			if key != "llama.block_count" {
				model = []GGUFMetadataKV{
					{Key: "general.architecture", ValueType: GGUFMetadataValueTypeString, Value: "clip"},
					{Key: "clip.projector_type", ValueType: GGUFMetadataValueTypeString, Value: "idefics3"},
				}
			}
			kv := GGUFMetadataKV{Key: key, ValueType: GGUFMetadataValueTypeUint32, Value: uint32(maxSaneBlockCount)}
			_, err := ParseGGUFFile(writeProjectorGGUF(t, append(model, kv)))
			require.NoError(t, err)
			kv.Value = uint32(maxSaneBlockCount + 1)
			_, err = ParseGGUFFile(writeProjectorGGUF(t, append(model, kv)))
			require.ErrorContains(t, err, key)
		})
	}
}

func TestArrayBudgetAcrossReads(t *testing.T) {
	var data bytes.Buffer
	for i := 0; i < 2; i++ {
		require.NoError(t, binary.Write(&data, binary.LittleEndian, GGUFMetadataValueTypeUint8))
		require.NoError(t, binary.Write(&data, binary.LittleEndian, uint64(2)))
		data.Write([]byte{1, 2})
	}
	budget := metadataReadBudget(64)
	rd := _GGUFReader{v: GGUFVersionV3, f: bytes.NewReader(data.Bytes()), bo: binary.LittleEndian, size: int64(data.Len()), budget: &budget}
	a, err := rd.ReadArray("samples")
	require.NoError(t, err)
	require.Equal(t, []any{uint8(1), uint8(2)}, a.Array)
	_, err = rd.ReadArray("samples")
	require.ErrorContains(t, err, "budget")
}

func TestRemoteShardLimitBeforeOpen(t *testing.T) {
	cli := &http.Client{Transport: httpx.RoundTripperFunc(func(*http.Request) (*http.Response, error) {
		t.Fatal("unexpected shard request")
		return nil, nil
	})}
	_, err := parseGGUFFileFromRemote(context.Background(), cli, "https://example.invalid/model-00001-of-02049.gguf", _GGUFReadOptions{})
	require.ErrorContains(t, err, "shard count")
}

func TestRemoteShardsShareDeadline(t *testing.T) {
	var file bytes.Buffer
	for _, v := range []any{GGUFMagicGGUFLe, uint32(GGUFVersionV3), uint64(0), uint64(0)} {
		require.NoError(t, binary.Write(&file, binary.LittleEndian, v))
	}
	file.Write(make([]byte, 8)) // Pad the header to the data alignment.
	var deadlines []time.Time
	var paths []string
	cli := &http.Client{Transport: httpx.RoundTripperFunc(func(req *http.Request) (*http.Response, error) {
		deadline, ok := req.Context().Deadline()
		require.True(t, ok)
		require.Equal(t, "download=true", req.URL.RawQuery)
		deadlines = append(deadlines, deadline)
		paths = append(paths, req.Method+" "+req.URL.Path)
		resp := &http.Response{Header: make(http.Header), StatusCode: http.StatusOK, ContentLength: int64(file.Len()), Body: http.NoBody}
		resp.Header.Set("Accept-Ranges", "bytes")
		if req.Method == http.MethodGet {
			resp.StatusCode = http.StatusPartialContent
			resp.Header.Set("Content-Range", "bytes 0-31/"+strconv.Itoa(file.Len()))
			resp.Body = io.NopCloser(bytes.NewReader(file.Bytes()))
		}
		return resp, nil
	})}
	_, err := parseGGUFFileFromRemote(context.Background(), cli, "https://example.invalid/model-00001-of-00002.gguf?download=true", _GGUFReadOptions{})
	require.NoError(t, err)
	require.Equal(t, []string{
		"HEAD /model-00001-of-00002.gguf", "GET /model-00001-of-00002.gguf",
		"HEAD /model-00002-of-00002.gguf", "GET /model-00002-of-00002.gguf",
	}, paths)
	for _, d := range deadlines {
		require.Equal(t, deadlines[0], d)
	}
}

func TestIndependentBlockCountMetadata(t *testing.T) {
	kv := GGUFMetadataKV{Key: "storage.block_count", ValueType: GGUFMetadataValueTypeUint32, Value: uint32(12000)}
	f, err := ParseGGUFFile(writeProjectorGGUF(t, []GGUFMetadataKV{kv}))
	require.NoError(t, err)
	got, ok := f.Header.MetadataKV.Get(kv.Key)
	require.True(t, ok)
	require.Equal(t, kv.Value, got.Value)
}

func TestMetadataReadBudgetOption(t *testing.T) {
	kv := GGUFMetadataKV{Key: "name", ValueType: GGUFMetadataValueTypeString, Value: "example"}
	file := writeProjectorGGUF(t, []GGUFMetadataKV{kv})
	_, err := ParseGGUFFile(file, UseMetadataReadBudget(100))
	require.ErrorContains(t, err, "budget")
	f, err := ParseGGUFFile(file, UseMetadataReadBudget(128))
	require.NoError(t, err)
	require.Equal(t, []GGUFMetadataKV{kv}, []GGUFMetadataKV(f.Header.MetadataKV))
}

func TestReadOptionsSelectCache(t *testing.T) {
	base := _GGUFReadOptions{CachePath: t.TempDir()}
	for _, opt := range []GGUFReadOption{UseMetadataReadBudget(512 << 20), UseRemoteShardLimit(512), SkipLargeMetadata()} {
		o := base
		opt(&o)
		require.NotEqual(t, base.parsedFileCache("remote"), o.parsedFileCache("remote"))
	}
}

func TestRemoteShardLimitOption(t *testing.T) {
	var o _GGUFReadOptions
	UseRemoteShardLimit(9999)(&o)
	files := ggufShardFiles("model-00002-of-09999.gguf")
	require.Equal(t, 9999, files.count)
	require.Equal(t, 9999, o.remoteShardLimit())
	require.Equal(t, "model-09999-of-09999.gguf", files.path(files.count-1))
}

func TestShardHeaderMatchesFilename(t *testing.T) {
	for _, tc := range []struct {
		key   string
		value uint32
	}{
		{"split.count", 3},
		{"split.no", 1},
	} {
		t.Run(tc.key, func(t *testing.T) {
			first := writeProjectorGGUF(t, []GGUFMetadataKV{{Key: tc.key, ValueType: GGUFMetadataValueTypeUint32, Value: tc.value}})
			named := filepath.Join(t.TempDir(), "model-00001-of-00002.gguf")
			require.NoError(t, os.Rename(first, named))
			_, err := ParseGGUFFile(named)
			require.ErrorContains(t, err, tc.key+" differs")
		})
	}
}

func TestRemoteReadDeadline(t *testing.T) {
	o := _GGUFReadOptions{}
	UseRemoteReadTimeout(30 * time.Minute)(&o)
	before := time.Now()
	ctx, cancel := o.remoteReadContext(context.Background())
	defer cancel()
	deadline, ok := ctx.Deadline()
	require.True(t, ok)
	require.WithinRange(t, deadline, before.Add(30*time.Minute), time.Now().Add(30*time.Minute))

	caller, stop := context.WithTimeout(context.Background(), time.Minute)
	defer stop()
	ctx, cancel = o.remoteReadContext(caller)
	defer cancel()
	parent, _ := caller.Deadline()
	deadline, _ = ctx.Deadline()
	require.Equal(t, parent, deadline)
}
