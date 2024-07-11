package main

import (
	"flag"
	"os"
	"fmt"
	"context"
	"strconv"
	"strings"
	"sync"

	"github.com/olekukonko/tablewriter"

	"github.com/thxcode/gguf-parser-go/util/anyx"
	"github.com/thxcode/gguf-parser-go/util/json"

	. "github.com/thxcode/gguf-parser-go"
	"regexp"
)

var Version = "v0.0.0"

func main() {
	ctx := context.Background()

	// Parse arguments.

	var (
		// model options
		path    string
		url     string
		hfRepo  string
		hfFile  string
		msRepo  string
		msFile  string
		olModel string
		olCrawl bool
		olUsage bool
		// read options
		debug         bool
		skipProxy     bool
		skipTLSVerify bool
		skipDNSCache  bool
		// estimate options
		ctxSize           = -1
		inMaxCtxSize      bool
		physicalBatchSize = 512
		parallelSize      = 1
		kvType            = "f16"
		noKVOffload       bool
		flashAttention    bool
		platformFootprint = "150,250"
		noMMap            bool
		offloadLayers     = -1
		offloadLayersStep uint64
		// output options
		version          bool
		raw              bool
		skipModel        bool
		skipArchitecture bool
		skipTokenizer    bool
		skipEstimate     bool
		inMib            bool
		inJson           bool
		inPrettyJson     = true
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
	fs.StringVar(&hfRepo, "repo", hfRepo, "[DEPRECATED, use --hf-repo instead] "+ // Deprecated, remove when release v0.3.0.
		"Repository of HuggingFace which the GGUF file store, e.g. "+
		"NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF, works with --file.")
	fs.StringVar(&hfFile, "file", hfFile, "[DEPRECATED, use --hf-file instead] "+ // Deprecated, remove when release v0.3.0.
		"Model file below the --repo, e.g. "+
		"Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.")
	fs.StringVar(&hfRepo, "hf-repo", hfRepo, "Repository of HuggingFace which the GGUF file store, e.g. "+
		"NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF, works with --hf-file.")
	fs.StringVar(&hfFile, "hf-file", hfFile, "Model file below the --hf-repo, e.g. "+
		"Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.")
	fs.StringVar(&msRepo, "ms-repo", msRepo, "Repository of ModelScope which the GGUF file store, e.g. "+
		"qwen/Qwen1.5-0.5B-Chat-GGUF, works with --ms-file.")
	fs.StringVar(&msFile, "ms-file", msFile, "Model file below the --ms-repo, e.g. "+
		"qwen1.5-0.5b-chat.gguf.")
	fs.StringVar(&olModel, "ol-model", olModel, "Model name of Ollama, e.g. "+
		"gemma2.")
	fs.BoolVar(&olCrawl, "ol-crawl", olCrawl, "Crawl the Ollama model instead of blobs fetching, "+
		"which will be more efficient and faster, but lossy.")
	fs.BoolVar(&olUsage, "ol-usage", olUsage, "Specify respecting the extending layers introduced by Ollama, "+
		"which affects the usage estimation.")
	fs.BoolVar(&debug, "debug", debug, "Enable debugging, verbosity.")
	fs.BoolVar(&skipProxy, "skip-proxy", skipProxy, "Skip proxy settings, "+
		"works with --url/--hf-*/--ol-*, "+
		"default is respecting the environment variables HTTP_PROXY/HTTPS_PROXY/NO_PROXY.")
	fs.BoolVar(&skipTLSVerify, "skip-tls-verify", skipTLSVerify, "Skip TLS verification, "+
		"works with --url/--hf-*/--ol-*, "+
		"default is verifying the TLS certificate on HTTPs request.")
	fs.BoolVar(&skipDNSCache, "skip-dns-cache", skipDNSCache, "Skip DNS cache, "+
		"works with --url/--hf-*/--ol-*, "+
		"default is caching the DNS lookup result.")
	fs.IntVar(&ctxSize, "ctx-size", ctxSize, "Specify the size of prompt context, "+
		"which is used to estimate the usage, "+
		"default is equal to the model's maximum context size.")
	fs.BoolVar(&inMaxCtxSize, "in-max-ctx-size", inMaxCtxSize, "Limit the context size to the maximum context size of the model, "+
		"if the context size is larger than the maximum context size.")
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
	fs.BoolVar(&noKVOffload, "no-kv-offload", noKVOffload, "Specify disabling Key-Value offloading, "+
		"which is used to estimate the usage. "+
		"Key-Value offloading can reduce the usage of VRAM.")
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
	fs.IntVar(&offloadLayers, "offload-layers", offloadLayers, "[DEPRECATED, use --gpu-layers instead] "+ // Deprecated, remove when release v0.3.0.
		"Specify how many layers to offload, "+
		"which is used to estimate the usage, "+
		"default is full offloaded.")
	fs.IntVar(&offloadLayers, "gpu-layers", offloadLayers, "Specify how many layers to offload, "+
		"which is used to estimate the usage, "+
		"default is full offloaded.")
	fs.Uint64Var(&offloadLayersStep, "offload-layers-step", offloadLayersStep, "[DEPRECATED, use --gpu-layers-step instead] "+ // Deprecated, remove when release v0.3.0.
		"Specify the step of layers to offload, "+
		"works with --offload-layers.")
	fs.Uint64Var(&offloadLayersStep, "gpu-layers-step", offloadLayersStep, "Specify the step of layers to offload, "+
		"works with --gpu-layers.")
	fs.BoolVar(&version, "version", version, "Show gguf-parser version.")
	fs.BoolVar(&raw, "raw", raw, "Output the file only, skip anything.")
	fs.BoolVar(&skipModel, "skip-model", skipModel, "Skip to display model metadata.")
	fs.BoolVar(&skipArchitecture, "skip-architecture", skipArchitecture, "Skip to display architecture metadata.")
	fs.BoolVar(&skipTokenizer, "skip-tokenizer", skipTokenizer, "Skip to display tokenizer metadata")
	fs.BoolVar(&skipEstimate, "skip-estimate", skipEstimate, "Skip to estimate.")
	fs.BoolVar(&inMib, "in-mib", inMib, "Display the estimated result in table with MiB.")
	fs.BoolVar(&inJson, "json", inJson, "Output as JSON.")
	fs.BoolVar(&inPrettyJson, "json-pretty", inPrettyJson, "Output as pretty JSON.")
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
	if skipTLSVerify {
		ropts = append(ropts, SkipTLSVerification())
	}
	if skipDNSCache {
		ropts = append(ropts, SkipDNSCache())
	}

	eopts := []LLaMACppUsageEstimateOption{
		WithCacheValueType(GGMLTypeF16),
		WithCacheKeyType(GGMLTypeF16),
	}
	if ctxSize > 0 {
		eopts = append(eopts, WithContextSize(int32(ctxSize)))
	}
	if inMaxCtxSize {
		eopts = append(eopts, WithinMaxContextSize())
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
	if noKVOffload {
		eopts = append(eopts, WithoutOffloadKVCache())
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
		case hfRepo != "" && hfFile != "":
			gf, err = ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfFile, ropts...)
		case msRepo != "" && msFile != "":
			gf, err = ParseGGUFFileFromModelScope(ctx, msRepo, msFile, ropts...)
		case olModel != "":
			om := ParseOllamaModel(olModel)
			gf, err = ParseGGUFFileFromOllamaModel(ctx, om, olCrawl, ropts...)
			if om != nil && olUsage {
				// Parameters override.
				{
					ps, _ := om.Params(ctx, nil)
					if v, ok := ps["num_ctx"]; ok {
						eopts = append(eopts, WithContextSize(anyx.Number[int32](v)))
					} else if ctxSize <= 0 {
						eopts = append(eopts, WithContextSize(2048))
					}
					if v, ok := ps["use_mmap"]; ok && !anyx.Bool(v) {
						noMMap = true
					}
					if v, ok := ps["num_gpu"]; ok {
						offloadLayers = anyx.Number[int](v)
					}
				}
				// Projector overlap,
				// in here, we just assume the projector is overlapped with its size to VRAM.
				{
					var sz uint64
					mls := om.SearchLayers(regexp.MustCompile(`^application/vnd\.ollama\.image\.projector$`))
					for i := range mls {
						sz += mls[i].Size
					}
					eopts = append(eopts, WithClipUsage(sz))
				}

			}
		}
		if err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "failed to parse GGUF file: %s\n", err.Error())
			os.Exit(1)
		}
	}

	// Output raw.

	if raw {
		enc := json.NewEncoder(os.Stdout)
		if inPrettyJson {
			enc.SetIndent("", "  ")
		}
		if err := enc.Encode(gf); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "failed to encode JSON: %s\n", err.Error())
			os.Exit(1)
		}
		return
	}

	// Otherwise, display the metadata and estimate the usage.

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

	// Then, output as JSON or table.

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

	if inJson {
		o := map[string]any{}
		if !skipModel {
			o["model"] = m
		}
		if !skipArchitecture {
			o["architecture"] = a
		}
		if !skipTokenizer && t.Model != "" {
			o["tokenizer"] = t
		}
		if !skipEstimate {
			es := e.Summarize(mmap, platformRAM, platformVRAM)
			if e.Architecture != "clip" {
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
			}
			o["estimate"] = es
		}

		enc := json.NewEncoder(os.Stdout)
		if inPrettyJson {
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
			[]string{"Name", "Arch", "Quantization", "Little Endian", "Size", "Parameters", "BPW"},
			nil,
			[]string{
				m.Name,
				m.Architecture,
				sprintf(m.FileType),
				sprintf(m.LittleEndian),
				sprintf(m.Size),
				sprintf(m.Parameters),
				sprintf(m.BitsPerWeight),
			})
	}

	if !skipArchitecture {
		var (
			hd []string
			bd []string
		)
		if a.Architecture != "clip" {
			hd = []string{"Max Context Len", "Embedding Len", "Embedding GQA", "Attention Head Cnt", "Layers", "Feed Forward Len", "Expert Cnt", "Vocabulary Len"}
			bd = []string{
				sprintf(a.MaximumContextLength),
				sprintf(a.EmbeddingLength),
				sprintf(a.EmbeddingGQA),
				sprintf(tenary(a.AttentionHeadCountKV == 0 || a.AttentionHeadCountKV == a.AttentionHeadCount, "N/A", a.AttentionHeadCount)),
				sprintf(a.BlockCount),
				sprintf(a.FeedForwardLength),
				sprintf(a.ExpertCount),
				sprintf(a.VocabularyLength),
			}
		} else {
			hd = []string{"Embedding Len", "Layers", "Feed Forward Len", "Encoder", "LLaVA Projector"}
			bd = []string{
				sprintf(a.EmbeddingLength),
				sprintf(a.BlockCount),
				sprintf(a.FeedForwardLength),
				sprintf(tenary(a.ClipHasTextEncoder, tenary(a.ClipHasVisionEncoder, "Text & Vision", "Text"), tenary(a.ClipHasVisionEncoder, "Vision", "N/A"))),
				sprintf(tenary(a.ClipHasLLaVaProjector, a.ClipProjectorType, "N/A")),
			}
		}
		tprint(
			"ARCHITECTURE",
			hd,
			nil,
			bd)
	}

	if !skipTokenizer && t.Model != "" {
		tprint(
			"TOKENIZER",
			[]string{"Model", "Tokens Size", "Tokens Len", "Added Tokens Len", "BOS Token", "EOS Token", "Unknown Token", "Separator Token", "Padding Token"},
			nil,
			[]string{
				t.Model,
				sprintf(tenary(t.TokensSize <= 0, "N/A", GGUFBytesScalar(t.TokensSize))),
				sprintf(tenary(t.TokensLength <= 0, "N/A", t.TokensLength)),
				sprintf(tenary(t.AddedTokensLength <= 0, "N/A", t.AddedTokensLength)),
				sprintf(tenary(t.BOSTokenID < 0, "N/A", t.BOSTokenID)),
				sprintf(tenary(t.EOSTokenID < 0, "N/A", t.EOSTokenID)),
				sprintf(tenary(t.UnknownTokenID < 0, "N/A", t.UnknownTokenID)),
				sprintf(tenary(t.SeparatorTokenID < 0, "N/A", t.SeparatorTokenID)),
				sprintf(tenary(t.PaddingTokenID < 0, "N/A", t.PaddingTokenID)),
			})
	}

	if !skipEstimate {
		var (
			hd  []string
			mg  []int
			bds [][]string
		)
		es := e.Summarize(mmap, platformRAM, platformVRAM)
		if e.Architecture != "clip" {
			hd = []string{"Arch", "Context Size", "Flash Attention", "MMap Support", "Offload Layers", "Full Offloaded", "UMA (RAM + VRAM)", "NonUMA RAM", "NonUMA VRAM"}
			mg = []int{0, 1, 2, 3, 5}

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

			bds = make([][]string, len(es.Memory))
			for i := range es.Memory {
				bds[i] = []string{
					sprintf(es.Architecture),
					sprintf(es.ContextSize),
					sprintf(es.FlashAttention),
					sprintf(!es.NoMMap),
					sprintf(tenary(es.Memory[i].FullOffloaded, sprintf("%d (%d + 1)", es.Memory[i].OffloadLayers, es.Memory[i].OffloadLayers-1), es.Memory[i].OffloadLayers)),
					sprintf(tenary(es.Memory[i].FullOffloaded, "Yes", "No")),
					sprintf("%s + %s = %s", es.Memory[i].UMA.RAM, es.Memory[i].UMA.VRAM, es.Memory[i].UMA.RAM+es.Memory[i].UMA.VRAM),
					sprintf(es.Memory[i].NonUMA.RAM),
					sprintf(es.Memory[i].NonUMA.VRAM),
				}
			}
		} else {
			hd = []string{"Arch", "Offload Layers", "Full Offloaded", "(V)RAM"}
			bds = [][]string{
				{
					sprintf(es.Architecture),
					sprintf(es.Memory[0].OffloadLayers),
					sprintf(tenary(es.Memory[0].FullOffloaded, "Yes", "No")),
					sprintf(max(es.Memory[0].UMA.RAM, es.Memory[0].UMA.VRAM)),
				},
			}
		}
		tprint(
			"ESTIMATE",
			hd,
			mg,
			bds...)
	}
}

func sprintf(f any, a ...any) string {
	if v, ok := f.(string); ok {
		if len(a) != 0 {
			return fmt.Sprintf(v, a...)
		}
		return v
	}
	return anyx.String(f)
}

func tprint(title string, header []string, merges []int, body ...[]string) {
	title = strings.ToUpper(title)

	tb := tablewriter.NewWriter(os.Stdout)

	tb.SetTablePadding("\t")
	tb.SetAlignment(tablewriter.ALIGN_CENTER)
	tb.SetHeaderLine(true)
	tb.SetRowLine(true)

	tb.SetHeaderAlignment(tablewriter.ALIGN_CENTER)
	tb.SetAutoFormatHeaders(false)
	tb.SetHeader(append([]string{"\\"}, header...))

	tb.SetAutoWrapText(false)
	tb.SetColMinWidth(0, 12)
	tb.SetAutoMergeCellsByColumnIndex(func() (r []int) {
		if len(merges) == 0 {
			return []int{0}
		}
		r = make([]int, len(merges)+1)
		r = append(r, 0)
		for i := range merges {
			r[i] = merges[i] + 1
		}
		return r
	}())
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
