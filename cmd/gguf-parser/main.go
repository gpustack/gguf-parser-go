package main

import (
	"flag"
	"os"
	"fmt"
	"context"
	"strconv"
	stdjson "encoding/json"

	"github.com/olekukonko/tablewriter"

	. "github.com/thxcode/gguf-parser-go"
)

var Version = "v0.0.0"

func main() {
	ctx := context.Background()

	// Parse arguments.

	var (
		// model
		path        string
		url         string
		repo, model string
		// read options
		debug       bool
		approximate = true
		mmap        = true
		skipProxy   bool
		skipTLS     bool
		// estimate options
		ctxSize = 512
		kvType  = "f16"
		// output
		json       bool
		jsonPretty = true
	)
	fs := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	fs.Usage = func() {
		_, _ = fmt.Fprintf(fs.Output(), "Usage of gguf-parser %v:\n", Version)
		fs.PrintDefaults()
	}
	fs.StringVar(&path, "path", path, "Path to load model, e.g. ~/.cache"+
		"/lm-studio/models/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/"+
		"Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf")
	fs.StringVar(&url, "url", url, "Url to load model, e.g. "+
		"https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF"+
		"/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf")
	fs.StringVar(&repo, "repo", repo, "Repo of HuggingFace, e.g. "+
		"NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF")
	fs.StringVar(&model, "model", model, "Model below the --repo, e.g. "+
		"Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf")
	fs.BoolVar(&debug, "debug", debug, "Debug mode")
	fs.BoolVar(&approximate, "approximate", approximate, "Speed up reading")
	fs.BoolVar(&mmap, "mmap", mmap, "Use mmap to read the local file")
	fs.BoolVar(&skipProxy, "skip-proxy", skipProxy, "Skip using proxy when reading from a remote URL")
	fs.BoolVar(&skipTLS, "skip-tls", skipTLS, "Skip TLS verification when reading from a remote URL")
	fs.IntVar(&ctxSize, "ctx-size", ctxSize, "Maximum context size to estimate memory usage")
	fs.StringVar(&kvType, "kv-type", kvType, "Key-Value cache type, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1]")
	fs.BoolVar(&json, "json", json, "Output as JSON")
	fs.BoolVar(&jsonPretty, "json-pretty", jsonPretty, "Output as pretty JSON")
	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	// Prepare options.

	var ropts []GGUFReadOption
	if approximate {
		ropts = append(ropts, UseApproximate())
	}
	if mmap {
		ropts = append(ropts, UseMMap())
	}
	if skipProxy {
		ropts = append(ropts, SkipProxy())
	}
	if skipTLS {
		ropts = append(ropts, SkipTLSVerification())
	}

	var eopts []GGUFEstimateOption
	if ctxSize > 0 {
		eopts = append(eopts, WithContextSize(int32(ctxSize)))
	}
	if kvType != "" {
		kv := GGMLTypeF16
		switch kvType {
		case "f32":
			kv = GGMLTypeF32
		case "f16":
			kv = GGMLTypeF16
		case "q8_0":
			kv = GGMLTypeQ8_0
		case "q4_0":
			kv = GGMLTypeQ4_0
		case "q4_1":
			kv = GGMLTypeQ4_1
		case "iq4_nl":
			kv = GGMLTypeIQ4_NL
		case "q5_0":
			kv = GGMLTypeQ5_0
		case "q5_1":
			kv = GGMLTypeQ5_1
		}
		eopts = append(eopts, WithCacheKeyType(kv), WithCacheValueType(kv))
	}

	// Parse GGUF file.

	var gf *GGUFFile
	{
		var err error
		switch {
		default:
			_, _ = fmt.Fprintf(os.Stderr, "no model specified\n")
			os.Exit(1)
		case path != "":
			gf, err = ParseGGUFFile(path, ropts...)
		case url != "":
			gf, err = ParseGGUFFileRemote(ctx, url, ropts...)
		case repo != "" && model != "":
			gf, err = ParseGGUFFileFromHuggingFace(ctx, repo, model, ropts...)
		}
		if err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "failed to parse GGUF file: %s\n", err.Error())
			os.Exit(1)
		}
	}

	m, a, e := gf.Model(), gf.Architecture(), gf.Estimate(eopts...)

	// Output

	if json {
		o := map[string]any{
			"model":        m,
			"architecture": a,
			"estimate":     e,
		}

		enc := stdjson.NewEncoder(os.Stdout)
		if jsonPretty {
			enc.SetIndent("", "  ")
		}
		if err := enc.Encode(o); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "failed to encode JSON: %s\n", err.Error())
			os.Exit(1)
		}

		return
	}

	tprintf(
		[]string{"Name", "Architecture", "Quantization Version", "File Type", "Little Endian", "Size", "Parameters", "BPW"},
		[]string{
			m.Name,
			m.Architecture,
			sprintf(m.QuantizationVersion),
			sprintf(m.FileType),
			sprintf(m.LittleEndian),
			m.Size.String(),
			m.Parameters.String(),
			m.BitsPerWeight.String(),
		})

	tprintf(
		[]string{"Context Length", "Embedding Length", "Layers", "Feed Forward Length", "Expert Count", "Vocabulary Length"},
		[]string{
			sprintf(a.ContextLength),
			sprintf(a.EmbeddingLength),
			fmt.Sprintf("%d + 1 = %d",
				a.BlockCount,
				a.BlockCount+1),
			sprintf(a.FeedForwardLength),
			sprintf(a.ExpertCount),
			sprintf(a.VocabularyLength),
		})

	tprintf(
		[]string{"Load Memory", "KVCache Memory", "Total Memory"},
		[]string{
			e.MemoryLoad.String(),
			e.KVCache.MemoryTotal.String(),
			e.MemoryTotal.String(),
		})
}

func sprintf(a any) string {
	switch v := a.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	case int:
		return strconv.Itoa(v)
	case int32:
		return strconv.Itoa(int(v))
	case int64:
		return strconv.Itoa(int(v))
	case uint:
		return strconv.Itoa(int(v))
	case uint32:
		return strconv.Itoa(int(v))
	case uint64:
		return strconv.Itoa(int(v))
	case float32:
		return strconv.FormatFloat(float64(v), 'f', -1, 32)
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func tprintf(headers, rows []string) {
	tb := tablewriter.NewWriter(os.Stdout)
	tb.SetHeaderAlignment(tablewriter.ALIGN_CENTER)
	tb.SetAlignment(tablewriter.ALIGN_CENTER)
	tb.SetHeaderLine(true)
	tb.SetBorder(true)
	tb.SetTablePadding("\t")
	tb.SetHeader(headers)
	tb.Append(rows)
	tb.Render()
	fmt.Println()
}
