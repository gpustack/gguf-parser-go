package main

import (
	"flag"
	"os"
	"fmt"
	"context"
	"strconv"
	"strings"
	"sync"
	stdjson "encoding/json"

	"github.com/olekukonko/tablewriter"

	. "github.com/thxcode/gguf-parser-go"
)

var Version = "v0.0.0"

func main() {
	ctx := context.Background()

	// Parse arguments.

	var (
		// model options
		path       string
		url        string
		repo, file string
		// read options
		debug         bool
		skipTLSVerify bool
		// estimate options
		ctxSize           = -1
		physicalBatchSize = 512
		parallelSize      = 1
		kvType            = "f16"
		flashAttention    bool
		platformFootprint = "150,250"
		noMMap            bool
		offloadLayers     = -1
		offloadLayersStep uint64
		// output options
		version          bool
		skipModel        bool
		skipArchitecture bool
		skipTokenizer    bool
		skipEstimate     bool
		inMib            bool
		json             bool
		jsonPretty       = true
	)
	fs := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	fs.Usage = func() {
		_, _ = fmt.Fprintf(fs.Output(), "Usage of gguf-parser %v:\n", Version)
		fs.PrintDefaults()
	}
	fs.StringVar(&path, "path", path, "Path where the GGUF file to load, e.g. ~/.cache"+
		"/lm-studio/models/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/"+
		"Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.")
	fs.StringVar(&url, "url", url, "Url where the GGUF file to load, e.g. "+
		"https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF"+
		"/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf. "+
		"Note that gguf-parser does not need to download the entire GGUF file.")
	fs.StringVar(&repo, "repo", repo, "Repository of HuggingFace which the GGUF file store, e.g. "+
		"NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF, works with --file. [Deprecated, use --hf-repo instead]")
	fs.StringVar(&file, "file", file, "Model file below the --repo, e.g. "+
		"Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf. [Deprecated, use --hf-file instead]") // Deprecated.
	fs.StringVar(&repo, "hf-repo", repo, "Repository of HuggingFace which the GGUF file store, e.g. "+
		"NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF, works with --hf-file.") // Deprecated.
	fs.StringVar(&file, "hf-file", file, "Model file below the --repo, e.g. "+
		"Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.")
	fs.BoolVar(&debug, "debug", debug, "Enable debugging, verbosity.")
	fs.BoolVar(&skipTLSVerify, "skip-tls-verify", skipTLSVerify, "Skip TLS verification, works with --url.")
	fs.IntVar(&ctxSize, "ctx-size", ctxSize, "Specify the size of prompt context, "+
		"which is used to estimate the usage, "+
		"default is equal to the model's maximum context size.")
	fs.IntVar(&physicalBatchSize, "ubatch-size", physicalBatchSize, "Specify the physical maximum batch size, "+
		"which is used to estimate the usage, "+
		"default is 512.")
	fs.IntVar(&parallelSize, "parallel-size", parallelSize, "Specify the number of parallel sequences to decode, "+
		"which is used to estimate the usage, "+
		"default is 1.")
	fs.StringVar(&kvType, "kv-type", kvType, "Specify the type of Key-Value cache, "+
		"which is used to estimate the usage, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1], "+
		"default is f16. "+
		"Use quantization type means enabling --flash-attention as well.")
	fs.BoolVar(&flashAttention, "flash-attention", flashAttention, "Specify enabling Flash Attention, "+
		"which is used to estimate the usage. "+
		"Flash Attention can reduce the usage of RAM/VRAM.")
	fs.StringVar(&platformFootprint, "platform-footprint", platformFootprint, "Specify the platform footprint(RAM,VRAM) in MiB, "+
		"which is used to estimate the NonUMA usage, "+
		"default is 150,250. "+
		"Different platform always gets different RAM and VRAM footprints, "+
		"for example, within CUDA, `cudaMemGetInfo` would occupy some RAM and VRAM, "+
		"see https://stackoverflow.com/questions/64854862/free-memory-occupied-by-cudamemgetinfo.")
	fs.BoolVar(&noMMap, "no-mmap", noMMap, "Specify disabling Memory-Mapped using, "+
		"which is used to estimate the usage. "+
		"Memory-Mapped can avoid loading the entire model weights into RAM.")
	fs.IntVar(&offloadLayers, "offload-layers", offloadLayers, "Specify how many layers to offload, "+
		"which is used to estimate the usage, "+
		"default is full offloaded. [Deprecated, use --gpu-layers instead]") // Deprecated.
	fs.IntVar(&offloadLayers, "gpu-layers", offloadLayers, "Specify how many layers to offload, "+
		"which is used to estimate the usage, "+
		"default is full offloaded.")
	fs.Uint64Var(&offloadLayersStep, "offload-layers-step", offloadLayersStep, "Specify the step of layers to offload, "+
		"works with --offload-layers. [Deprecated, use --gpu-layers-step instead]") // Deprecated.
	fs.Uint64Var(&offloadLayersStep, "gpu-layers-step", offloadLayersStep, "Specify the step of layers to offload, "+
		"works with --gpu-layers.")
	fs.BoolVar(&version, "version", version, "Show gguf-parser version.")
	fs.BoolVar(&skipModel, "skip-model", skipModel, "Skip to display model metadata.")
	fs.BoolVar(&skipArchitecture, "skip-architecture", skipArchitecture, "Skip to display architecture metadata.")
	fs.BoolVar(&skipTokenizer, "skip-tokenizer", skipTokenizer, "Skip to display tokenizer metadata")
	fs.BoolVar(&skipEstimate, "skip-estimate", skipEstimate, "Skip to estimate.")
	fs.BoolVar(&inMib, "in-mib", inMib, "Display the estimated result in table with MiB.")
	fs.BoolVar(&json, "json", json, "Output as JSON.")
	fs.BoolVar(&jsonPretty, "json-pretty", jsonPretty, "Output as pretty JSON.")
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
	if skipTLSVerify {
		ropts = append(ropts, SkipTLSVerification())
	}

	eopts := []LLaMACppUsageEstimateOption{
		WithCacheValueType(GGMLTypeF16),
		WithCacheKeyType(GGMLTypeF16),
	}
	if ctxSize > 0 {
		eopts = append(eopts, WithContextSize(int32(ctxSize)))
	}
	if physicalBatchSize > 0 {
		eopts = append(eopts, WithPhysicalBatchSize(int32(physicalBatchSize)))
	}
	if parallelSize > 0 {
		eopts = append(eopts, WithParallelSize(int32(parallelSize)))
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
	if flashAttention {
		eopts = append(eopts, WithFlashAttention())
	}

	// Parse GGUF file.

	var gf *GGUFFile
	{
		ropts := ropts[:len(ropts):len(ropts)]

		var err error
		switch {
		default:
			_, _ = fmt.Fprintf(os.Stderr, "no model specified\n")
			os.Exit(1)
		case path != "":
			gf, err = ParseGGUFFile(path, ropts...)
		case url != "":
			gf, err = ParseGGUFFileRemote(ctx, url, ropts...)
		case repo != "" && file != "":
			gf, err = ParseGGUFFileFromHuggingFace(ctx, repo, file, ropts...)
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
	if !skipArchitecture && !skipEstimate {
		a = gf.Architecture()
	}
	if !skipTokenizer && !skipEstimate {
		t = gf.Tokenizer()
	}
	if !skipEstimate {
		eopts := eopts[:len(eopts):len(eopts)]

		if offloadLayers >= 0 {
			eopts = append(eopts, WithOffloadLayers(uint64(offloadLayers)))
		}
		e = gf.EstimateLLaMACppUsage(eopts...)
	}

	// Output
	var (
		mmap                      = !noMMap
		platformRAM, platformVRAM uint64
	)
	{
		if platformFootprint != "" {
			parts := strings.Split(platformFootprint, ",")
			if len(parts) == 2 {
				if v, err := strconv.ParseUint(parts[0], 10, 64); err == nil {
					platformRAM = v * 1024 * 1024
				}
				if v, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
					platformVRAM = v * 1024 * 1024
				}
			}
		}
	}

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
			es := e.Summarize(mmap, platformRAM, platformVRAM)
			switch {
			case offloadLayersStep > e.OffloadLayers:
				offloadLayersStep = e.OffloadLayers
			case offloadLayersStep <= 0:
				offloadLayersStep = e.OffloadLayers
			}
			if offloadLayersStep < e.OffloadLayers {
				cnt := e.OffloadLayers/offloadLayersStep + 1
				if e.OffloadLayers%offloadLayersStep != 0 {
					cnt++
				}
				ess := make([]LLaMACppUsageEstimateMemorySummary, cnt)
				var wg sync.WaitGroup
				for i := 0; i < cap(ess); i++ {
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						eopts := eopts[:len(eopts):len(eopts)]
						eopts = append(eopts, WithOffloadLayers(uint64(i)*offloadLayersStep))
						ess[i] = gf.EstimateLLaMACppUsage(eopts...).SummarizeMemory(mmap, platformRAM, platformVRAM)
					}(i)
				}
				wg.Wait()
				ess[cap(ess)-1] = es.Memory[0]
				es.Memory = ess
			}
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

	InMiBytes = inMib

	if !skipModel {
		tprint(
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
		tprint(
			"ARCHITECTURE",
			[]string{"Max Context Len", "Embedding Len", "Embedding GQA", "Attention Head Cnt", "Layers", "Feed Forward Len", "Expert Cnt", "Vocabulary Len"},
			[]string{
				sprintf(a.MaximumContextLength),
				sprintf(a.EmbeddingLength),
				sprintf(a.EmbeddingGQA),
				sprintf(tenary(a.AttentionHeadCountKV == 0 || a.AttentionHeadCountKV == a.AttentionHeadCount, "N/A", a.AttentionHeadCount)),
				sprintf(a.BlockCount),
				sprintf(a.FeedForwardLength),
				sprintf(a.ExpertCount),
				sprintf(a.VocabularyLength),
			})
	}

	if !skipTokenizer {
		tprint(
			"TOKENIZER",
			[]string{"Model", "Tokens Size", "Tokens Len", "Added Tokens Len", "BOS Token", "EOS Token", "Unknown Token", "Separator Token", "Padding Token"},
			[]string{
				t.Model,
				sprintf(GGUFBytesScalar(t.TokensSize)),
				sprintf(t.TokensLength),
				sprintf(t.AddedTokensLength),
				sprintf(tenary(t.BOSTokenID < 0, "N/A", t.BOSTokenID)),
				sprintf(tenary(t.EOSTokenID < 0, "N/A", t.EOSTokenID)),
				sprintf(tenary(t.UnknownTokenID < 0, "N/A", t.UnknownTokenID)),
				sprintf(tenary(t.SeparatorTokenID < 0, "N/A", t.SeparatorTokenID)),
				sprintf(tenary(t.PaddingTokenID < 0, "N/A", t.PaddingTokenID)),
			})
	}

	if !skipEstimate {
		es := e.Summarize(mmap, platformRAM, platformVRAM)
		switch {
		case offloadLayersStep > e.OffloadLayers:
			offloadLayersStep = e.OffloadLayers
		case offloadLayersStep <= 0:
			offloadLayersStep = e.OffloadLayers
		}
		if offloadLayersStep < e.OffloadLayers {
			cnt := e.OffloadLayers/offloadLayersStep + 1
			if e.OffloadLayers%offloadLayersStep != 0 || e.FullOffloaded {
				cnt++
			}
			ess := make([]LLaMACppUsageEstimateMemorySummary, cnt)
			var wg sync.WaitGroup
			for i := 0; i < cap(ess); i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					eopts := eopts[:len(eopts):len(eopts)]
					eopts = append(eopts, WithOffloadLayers(uint64(i)*offloadLayersStep))
					ess[i] = gf.EstimateLLaMACppUsage(eopts...).SummarizeMemory(mmap, platformRAM, platformVRAM)
				}(i)
			}
			wg.Wait()
			ess[cap(ess)-1] = es.Memory[0]
			es.Memory = ess
		}
		bd := make([][]string, len(es.Memory))
		for i := range es.Memory {
			bd[i] = []string{
				sprintf(es.Architecture),
				sprintf(es.ContextSize),
				sprintf(es.FlashAttention),
				sprintf(!es.NoMMap),
				sprintf(tenary(es.Memory[i].FullOffloaded, sprintf("%d (%d + 1)", es.Memory[i].OffloadLayers, es.Memory[i].OffloadLayers-1), es.Memory[i].OffloadLayers)),
				sprintf(tenary(es.Memory[i].FullOffloaded, "Yes", "No")),
				sprintf(es.Memory[i].UMA),
				sprintf(es.Memory[i].NonUMA.RAM),
				sprintf(es.Memory[i].NonUMA.VRAM),
			}
		}
		tprint(
			"ESTIMATE",
			[]string{"Arch", "Context Size", "Flash Attention", "MMap Support", "Offload Layers", "Full Offloaded", "UMA RAM", "NonUMA RAM", "NonUMA VRAM"},
			bd...)
	}
}

func sprintf(f any, a ...any) string {
	switch v := f.(type) {
	case string:
		if len(a) != 0 {
			return fmt.Sprintf(v, a...)
		}
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

func tprint(title string, header []string, body ...[]string) {
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
	for i := range body {
		tb.Append(append([]string{title}, body[i]...))
	}
	tb.Render()
	fmt.Println()
}

func tenary(c bool, t, f any) any {
	if c {
		return t
	}
	return f
}
