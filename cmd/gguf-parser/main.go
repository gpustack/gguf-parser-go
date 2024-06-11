package main

import (
	"flag"
	"os"
	"fmt"
	"context"
	"strconv"
	"strings"
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
		debug     bool
		skipProxy bool
		skipTLS   bool
		// estimate options
		ctxSize       = -1
		kvType        = "f16"
		offloadLayers = -1
		batchSize     = 512
		parallel      = 1
		noMMap        bool
		// output options
		version          bool
		skipModel        bool
		skipArchitecture bool
		skipTokenizer    bool
		skipEstimate     bool
		json             bool
		jsonPretty       = true
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
	fs.BoolVar(&skipProxy, "skip-proxy", skipProxy, "Skip using proxy when reading from a remote URL")
	fs.BoolVar(&skipTLS, "skip-tls", skipTLS, "Skip TLS verification when reading from a remote URL")
	fs.IntVar(&ctxSize, "ctx-size", ctxSize, "Context size to estimate memory usage, default is equal to the model's maximum context size")
	fs.StringVar(&kvType, "kv-type", kvType, "Key-Value cache type, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1]")
	fs.IntVar(&offloadLayers, "offload-layers", offloadLayers, "Specify how many layers to offload, default is fully offloading")
	fs.IntVar(&batchSize, "batch-size", batchSize, "Physical maximum batch size")
	fs.IntVar(&parallel, "parallel", parallel, "Number of parallel sequences to decode")
	fs.BoolVar(&noMMap, "no-mmap", noMMap, "Do not use memory-mapping, which influences the estimate result")
	fs.BoolVar(&version, "version", version, "Show version")
	fs.BoolVar(&skipModel, "skip-model", skipModel, "Skip model metadata")
	fs.BoolVar(&skipArchitecture, "skip-architecture", skipArchitecture, "Skip architecture metadata")
	fs.BoolVar(&skipTokenizer, "skip-tokenizer", skipTokenizer, "Skip tokenizer metadata")
	fs.BoolVar(&skipEstimate, "skip-estimate", skipEstimate, "Skip estimate")
	fs.BoolVar(&json, "json", json, "Output as JSON")
	fs.BoolVar(&jsonPretty, "json-pretty", jsonPretty, "Output as pretty JSON")
	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	if version {
		fmt.Printf("gguf-parser %s\n", Version)
		return
	}

	// Prepare options.

	ropts := []GGUFReadOption{
		SkipLargeMetadata(),
		UseMMap(),
	}
	if debug {
		ropts = append(ropts, UseDebug())
	}
	if skipProxy {
		ropts = append(ropts, SkipProxy())
	}
	if skipTLS {
		ropts = append(ropts, SkipTLSVerification())
	}

	eopts := []LLaMACppUsageEstimateOption{
		WithCacheValueType(GGMLTypeF16),
		WithCacheKeyType(GGMLTypeF16),
	}
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
	if offloadLayers >= 0 {
		eopts = append(eopts, WithOffloadLayers(uint64(offloadLayers)))
	}
	if batchSize > 0 {
		eopts = append(eopts, WithBatchSize(int32(batchSize)))
	}
	if parallel > 0 {
		eopts = append(eopts, WithParallelSize(int32(parallel)))
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

	var (
		m GGUFModelMetadata
		a GGUFArchitectureMetadata
		t GGUFTokenizerMetadata
		e LLaMACppUsageEstimate
	)
	if !skipModel {
		m = gf.Model()
	}
	if !skipArchitecture {
		a = gf.Architecture()
	}
	if !skipTokenizer {
		t = gf.Tokenizer()
	}
	if !skipEstimate {
		e = gf.EstimateLLaMACppUsage(eopts...)
	}

	// Output

	if json {
		o := map[string]any{}
		if !skipModel {
			o["model"] = m
		}
		if !skipArchitecture {
			o["architecture"] = a
		}
		if !skipTokenizer {
			o["tokenizer"] = t
		}
		if !skipEstimate {
			es := e.Summarize(!noMMap)
			o["estimate"] = es
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

	if !skipModel {
		tprintf(
			"MODEL",
			[]string{"Name", "Arch", "Quantization Version", "File Type", "Little Endian", "Size", "Parameters", "BPW"},
			[]string{
				m.Name,
				m.Architecture,
				sprintf(m.QuantizationVersion),
				sprintf(m.FileType),
				sprintf(m.LittleEndian),
				sprintf(m.Size),
				sprintf(m.Parameters),
				sprintf(m.BitsPerWeight),
			})
	}

	if !skipArchitecture {
		tprintf(
			"ARCHITECTURE",
			[]string{"Max Context Len", "Embedding Len", "Layers", "Feed Forward Len", "Expert Cnt", "Vocabulary Len"},
			[]string{
				sprintf(a.MaximumContextLength),
				sprintf(a.EmbeddingLength),
				sprintf(a.BlockCount),
				sprintf(a.FeedForwardLength),
				sprintf(a.ExpertCount),
				sprintf(a.VocabularyLength),
			})
	}

	if !skipTokenizer {
		sprintTokenID := func(a int64) string {
			if a < 0 {
				return "N/A"
			}
			return sprintf(a)
		}
		tprintf(
			"TOKENIZER",
			[]string{"Model", "Tokens Size", "Tokens Len", "Added Tokens Len", "BOS Token", "EOS Token", "Unknown Token", "Separator Token", "Padding Token"},
			[]string{
				t.Model,
				sprintf(GGUFBytesScalar(t.TokensSize)),
				sprintf(t.TokensLength),
				sprintf(t.AddedTokensLength),
				sprintTokenID(t.BOSTokenID),
				sprintTokenID(t.EOSTokenID),
				sprintTokenID(t.UnknownTokenID),
				sprintTokenID(t.SeparatorTokenID),
				sprintTokenID(t.PaddingTokenID),
			})
	}

	if !skipEstimate {
		es := e.Summarize(!noMMap)
		if ctxSize <= 0 {
			if a.MaximumContextLength == 0 {
				a = gf.Architecture()
			}
			ctxSize = int(a.MaximumContextLength)
		}
		tprintf(
			"ESTIMATE",
			[]string{"Mem. Arch", "MMap", "Context Size", "Usage"},
			[]string{
				"UMA",
				sprintf(!noMMap),
				sprintf(ctxSize),
				sprintf(es.UMA),
			},
			[]string{
				"NonUMA",
				sprintf(!noMMap),
				sprintf(ctxSize),
				fmt.Sprintf("%s(RAM) + %s (VRAM)", es.NonUMA.RAM, es.NonUMA.VRAM),
			})
	}
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

func tprintf(title string, header []string, body ...[]string) {
	title = strings.ToUpper(title)
	for i := range header {
		header[i] = strings.ToUpper(header[i])
	}

	tb := tablewriter.NewWriter(os.Stdout)
	tb.SetTablePadding("\t")
	tb.SetAlignment(tablewriter.ALIGN_CENTER)
	tb.SetHeaderLine(true)
	tb.SetRowLine(true)
	tb.SetAutoMergeCellsByColumnIndex([]int{0, 1, 2, 3})
	tb.Append(append([]string{title}, header...))
	for i := range body {
		tb.Append(append([]string{title}, body[i]...))
	}
	tb.Render()
	fmt.Println()
}
