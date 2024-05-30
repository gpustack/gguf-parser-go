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
		mmap      = true
		skipProxy bool
		skipTLS   bool
		// estimate options
		ctxSize       = 512
		kvType        = "f16"
		offloadLayers uint64
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
	fs.BoolVar(&mmap, "mmap", mmap, "Use mmap to read the local file")
	fs.BoolVar(&skipProxy, "skip-proxy", skipProxy, "Skip using proxy when reading from a remote URL")
	fs.BoolVar(&skipTLS, "skip-tls", skipTLS, "Skip TLS verification when reading from a remote URL")
	fs.IntVar(&ctxSize, "ctx-size", ctxSize, "Context size to estimate memory usage")
	fs.StringVar(&kvType, "kv-type", kvType, "Key-Value cache type, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1]")
	fs.Uint64Var(&offloadLayers, "offload-layers", offloadLayers, "Specify how many layers to offload, default is fully offloading")
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
	}
	if debug {
		ropts = append(ropts, UseDebug())
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

	if ctxSize <= 0 {
		ctxSize = 512
	}
	eopts := []GGUFEstimateOption{
		WithContextSize(int32(ctxSize)),
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
	if offloadLayers > 0 {
		eopts = append(eopts, WithOffloadLayers(offloadLayers))
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
		e GGUFEstimate
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
		e = gf.Estimate(eopts...)
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
			o["estimate"] = e
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
	}

	if !skipArchitecture {
		tprintf(
			"ARCHITECTURE",
			[]string{"Max Context Length", "Embedding Length", "Layers", "Feed Forward Length", "Expert Count", "Vocabulary Length"},
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
			[]string{"Model", "Tokens Length", "Added Tokens Length", "BOS Token", "EOS Token", "Unknown Token", "Separator Token", "Padding Token"},
			[]string{
				t.Model,
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
		tprintf(
			"ESTIMATE TOTAL",
			[]string{"Context Length", "KV Cache", "Compute Memory", "IO Memory", "Sum"},
			[]string{
				sprintf(ctxSize),
				e.Total.KVCache.Sum().String(),
				e.Total.Compute.String(),
				e.Total.IO.String(),
				e.Offload.Sum().String(),
			})
		if e.Offload != nil {
			tprintf(
				"ESTIMATE OFFLOAD",
				[]string{"Context Length", "KV Cache", "Compute Memory", "IO Memory", "Sum"},
				[]string{
					sprintf(ctxSize),
					e.Offload.KVCache.Sum().String(),
					e.Offload.Compute.String(),
					e.Offload.IO.String(),
					e.Offload.Sum().String(),
				})
		}
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

func tprintf(title string, header, body []string) {
	title = strings.ToUpper(title)
	for i := range header {
		header[i] = strings.ToUpper(header[i])
	}

	tb := tablewriter.NewWriter(os.Stdout)
	tb.SetTablePadding("\t")
	tb.SetAlignment(tablewriter.ALIGN_CENTER)
	tb.SetHeaderLine(true)
	tb.SetRowLine(true)
	tb.SetAutoMergeCells(true)
	tb.Append(append([]string{title}, header...))
	tb.Append(append([]string{title}, body...))
	tb.Render()
	fmt.Println()
}
