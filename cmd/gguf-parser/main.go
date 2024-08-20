package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/gpustack/gguf-parser-go/util/anyx"
	"github.com/gpustack/gguf-parser-go/util/json"
	"github.com/gpustack/gguf-parser-go/util/osx"
	"github.com/gpustack/gguf-parser-go/util/signalx"
	"github.com/jedib0t/go-pretty/v6/table"
	"github.com/jedib0t/go-pretty/v6/text"
	"github.com/urfave/cli/v2"

	. "github.com/gpustack/gguf-parser-go" // nolint: stylecheck
)

var Version = "v0.0.0"

func init() {
	cli.VersionFlag = &cli.BoolFlag{
		Name:               "version",
		Aliases:            []string{"v"},
		Usage:              "Print the version.",
		DisableDefaultText: true,
	}
	cli.HelpFlag = &cli.BoolFlag{
		Name:               "help",
		Aliases:            []string{"h"},
		Usage:              "Print the usage.",
		DisableDefaultText: true,
	}
}

func main() {
	name := filepath.Base(os.Args[0])
	app := &cli.App{
		Name:            name,
		Usage:           "Review/Check/Estimate the GGUF file.",
		UsageText:       name + " [GLOBAL OPTIONS]",
		Version:         Version,
		Reader:          os.Stdin,
		Writer:          os.Stdout,
		ErrWriter:       os.Stderr,
		HideHelpCommand: true,
		OnUsageError: func(c *cli.Context, _ error, _ bool) error {
			return cli.ShowAppHelp(c)
		},
		Flags: []cli.Flag{
			&cli.BoolFlag{
				Destination: &debug,
				Value:       debug,
				Name:        "debug",
				Usage:       "Enable debugging, verbosity.",
			},
			&cli.StringFlag{
				Destination: &path,
				Value:       path,
				Category:    "Model/Local",
				Name:        "path",
				Aliases:     []string{"model", "m"},
				Usage: "Path where the GGUF file to load for the main model, e.g. ~/.cache" +
					"/lm-studio/models/QuantFactory/Qwen2-7B-Instruct-GGUF" +
					"/Qwen2-7B-Instruct.Q5_K_M.gguf.",
			},
			&cli.StringFlag{
				Destination: &draftPath,
				Value:       draftPath,
				Category:    "Model/Local",
				Name:        "draft-path",
				Aliases:     []string{"model-draft", "md"},
				Usage: "Path where the GGUF file to load for the draft model, optional, e.g. ~/.cache" +
					"/lm-studio/models/QuantFactory/Qwen2-1.5B-Instruct-GGUF" +
					"/Qwen2-1.5B-Instruct.Q5_K_M.gguf",
			},
			&cli.StringFlag{
				Destination: &mmprojPath,
				Value:       mmprojPath,
				Category:    "Model/Local",
				Name:        "mmproj-path",
				Aliases:     []string{"mmproj"},
				Usage:       "Path where the GGUF file to load for the multimodal projector, optional.",
			},
			&cli.StringFlag{
				Destination: &url,
				Value:       url,
				Category:    "Model/Remote",
				Name:        "url",
				Aliases:     []string{"model-url", "mu"},
				Usage: "Url where the GGUF file to load for the main model, e.g. " +
					"https://huggingface.co/QuantFactory/Qwen2-7B-Instruct-GGUF" +
					"/resolve/main/Qwen2-7B-Instruct.Q5_K_M.gguf. " +
					"Note that gguf-parser does not need to download the entire GGUF file.",
			},
			&cli.StringFlag{
				Destination: &draftUrl,
				Value:       draftUrl,
				Category:    "Model/Remote",
				Name:        "draft-url",
				Usage: "Url where the GGUF file to load for the draft model, optional, e.g. " +
					"https://huggingface.co/QuantFactory/Qwen2-1.5B-Instruct-GGUF" +
					"/resolve/main/Qwen2-1.5B-Instruct.Q5_K_M.gguf. " +
					"Note that gguf-parser does not need to download the entire GGUF file.",
			},
			&cli.StringFlag{
				Destination: &mmprojUrl,
				Value:       mmprojUrl,
				Category:    "Model/Remote",
				Name:        "mmproj-url",
				Usage:       "Url where the GGUF file to load for the multimodal projector, optional.",
			},
			&cli.StringFlag{
				Destination: &token,
				Value:       token,
				Category:    "Model/Remote",
				Name:        "token",
				Usage: "Bearer auth token to load GGUF file, optional, " +
					"works with --url/--draft-url.",
			},
			&cli.StringFlag{
				Destination: &hfRepo,
				Value:       hfRepo,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-repo",
				Aliases:     []string{"hfr"},
				Usage: "Repository of HuggingFace which the GGUF file store for the main model, e.g. " +
					"QuantFactory/Qwen2-7B-Instruct-GGUF, works with --hf-file.",
			},
			&cli.StringFlag{
				Destination: &hfFile,
				Value:       hfFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-file",
				Aliases:     []string{"hff"},
				Usage: "Model file below the --hf-repo, e.g. " +
					"Qwen2-7B-Instruct.Q5_K_M.gguf.",
			},
			&cli.StringFlag{
				Destination: &hfMMProjFile,
				Value:       hfMMProjFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-mmproj-file",
				Usage:       "Multimodal projector file below the --hf-repo.",
			},
			&cli.StringFlag{
				Destination: &hfDraftRepo,
				Value:       hfDraftRepo,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-draft-repo",
				Usage: "Repository of HuggingFace which the GGUF file store for the draft model, optional, e.g. " +
					"QuantFactory/Qwen2-1.5B-Instruct-GGUF, works with --hf-draft-file.",
			},
			&cli.StringFlag{
				Destination: &hfDraftFile,
				Value:       hfDraftFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-draft-file",
				Usage: "Model file below the --hf-draft-repo, optional, e.g. " +
					"Qwen2-1.5B-Instruct.Q5_K_M.gguf.",
			},
			&cli.StringFlag{
				Destination: &hfToken,
				Value:       hfToken,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-token",
				Aliases:     []string{"hft"},
				Usage: "User access token of HuggingFace, optional, " +
					"works with --hf-repo/--hf-file pair or --hf-draft-repo/--hf-draft-file pair. " +
					"See https://huggingface.co/settings/tokens.",
			},
			&cli.StringFlag{
				Destination: &msRepo,
				Value:       msRepo,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-repo",
				Usage: "Repository of ModelScope which the GGUF file store for the main model, e.g. " +
					"qwen/Qwen1.5-7B-Chat-GGUF, works with --ms-file.",
			},
			&cli.StringFlag{
				Destination: &msFile,
				Value:       msFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-file",
				Usage: "Model file below the --ms-repo, e.g. " +
					"qwen1_5-7b-chat-q5_k_m.gguf.",
			},
			&cli.StringFlag{
				Destination: &msMMProjFile,
				Value:       msMMProjFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-mmproj-file",
				Usage:       "Multimodal projector file below the --ms-repo.",
			},
			&cli.StringFlag{
				Destination: &msDraftRepo,
				Value:       msDraftRepo,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-draft-repo",
				Usage: "Repository of ModelScope which the GGUF file store for the draft model, optional, e.g. " +
					"qwen/Qwen1.5-1.8B-Chat-GGUF, works with --ms-draft-file.",
			},
			&cli.StringFlag{
				Destination: &msDraftFile,
				Value:       msDraftFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-draft-file",
				Usage: "Model file below the --ms-draft-repo, optional, e.g. " +
					"qwen1_5-1_8b-chat-q5_k_m.gguf.",
			},
			&cli.StringFlag{
				Destination: &msToken,
				Value:       msToken,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-token",
				Usage: "Git access token of ModelScope, optional, " +
					"works with --ms-repo/--ms-file pair or --ms-draft-repo/--ms-draft-file pair. " +
					"See https://modelscope.cn/my/myaccesstoken.",
			},
			&cli.StringFlag{
				Destination: &olModel,
				Value:       olModel,
				Category:    "Model/Remote/Ollama",
				Name:        "ol-model",
				Usage: "Model name of Ollama, e.g. " +
					"gemma2.",
			},
			&cli.BoolFlag{
				Destination: &olUsage,
				Value:       olUsage,
				Category:    "Model/Remote/Ollama",
				Name:        "ol-usage",
				Usage: "Specify respecting the extending layers introduced by Ollama, " +
					"works with --ol-model, which affects the usage estimation.",
			},
			&cli.BoolFlag{
				Destination: &skipProxy,
				Value:       skipProxy,
				Category:    "Load",
				Name:        "skip-proxy",
				Usage: "Skip proxy settings, " +
					"works with --url/--hf-*/--ms-*/--ol-*, " +
					"default is respecting the environment variables HTTP_PROXY/HTTPS_PROXY/NO_PROXY.",
			},
			&cli.BoolFlag{
				Destination: &skipTLSVerify,
				Value:       skipTLSVerify,
				Category:    "Load",
				Name:        "skip-tls-verify",
				Usage: "Skip TLS verification, " +
					"works with --url/--hf-*/--ms-*/--ol-*, " +
					"default is verifying the TLS certificate on HTTPs request.",
			},
			&cli.BoolFlag{
				Destination: &skipDNSCache,
				Value:       skipDNSCache,
				Category:    "Load",
				Name:        "skip-dns-cache",
				Usage: "Skip DNS cache, " +
					"works with --url/--hf-*/--ms-*/--ol-*, " +
					"default is caching the DNS lookup result.",
			},
			&cli.BoolFlag{
				Destination: &skipRangDownloadDetect,
				Value:       skipRangDownloadDetect,
				Category:    "Load",
				Name:        "skip-rang-download-detect",
				Usage: "Skip range download detect, " +
					"works with --url/--hf-*/--ms-*/--ol-*, " +
					"default is detecting the range download support.",
			},
			&cli.BoolFlag{
				Destination: &skipCache,
				Value:       skipCache,
				Category:    "Load",
				Name:        "skip-cache",
				Usage: "Skip cache, " +
					"works with --url/--hf-*/--ms-*/--ol-*, " +
					"default is caching the read result.",
			},
			&cli.IntFlag{
				Destination: &ctxSize,
				Value:       ctxSize,
				Category:    "Estimate",
				Name:        "ctx-size",
				Aliases:     []string{"c"},
				Usage: "Specify the size of prompt context, " +
					"which is used to estimate the usage, " +
					"default is equal to the model's maximum context size.",
			},
			&cli.BoolFlag{
				Destination: &inMaxCtxSize,
				Value:       inMaxCtxSize,
				Category:    "Estimate",
				Name:        "in-max-ctx-size",
				Usage: "Limit the context size to the maximum context size of the model, " +
					"if the context size is larger than the maximum context size.",
			},
			&cli.IntFlag{
				Destination: &logicalBatchSize,
				Value:       logicalBatchSize,
				Category:    "Estimate",
				Name:        "batch-size",
				Aliases:     []string{"b"},
				Usage: "Specify the logical batch size, " +
					"which is used to estimate the usage.",
			},
			&cli.IntFlag{
				Destination: &physicalBatchSize,
				Value:       physicalBatchSize,
				Category:    "Estimate",
				Name:        "ubatch-size",
				Aliases:     []string{"ub"},
				Usage: "Specify the physical maximum batch size, " +
					"which is used to estimate the usage.",
			},
			&cli.IntFlag{
				Destination: &parallelSize,
				Value:       parallelSize,
				Category:    "Estimate",
				Name:        "parallel-size",
				Aliases:     []string{"parallel", "np"},
				Usage: "Specify the number of parallel sequences to decode, " +
					"which is used to estimate the usage.",
			},
			&cli.StringFlag{
				Destination: &cacheKeyType,
				Value:       cacheKeyType,
				Category:    "Estimate",
				Name:        "cache-type-k",
				Aliases:     []string{"ctk"},
				Usage: "Specify the type of Key cache, " +
					"which is used to estimate the usage, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1].",
			},
			&cli.StringFlag{
				Destination: &cacheValueType,
				Value:       cacheValueType,
				Category:    "Estimate",
				Name:        "cache-type-v",
				Aliases:     []string{"ctv"},
				Usage: "Specify the type of Value cache, " +
					"which is used to estimate the usage, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1].",
			},
			&cli.BoolFlag{
				Destination: &noKVOffload,
				Value:       noKVOffload,
				Category:    "Estimate",
				Name:        "no-kv-offload",
				Aliases:     []string{"nkvo"},
				Usage: "Specify disabling Key-Value offloading, " +
					"which is used to estimate the usage. " +
					"Disable Key-Value offloading can reduce the usage of VRAM.",
			},
			&cli.BoolFlag{
				Destination: &flashAttention,
				Value:       flashAttention,
				Category:    "Estimate",
				Name:        "flash-attention",
				Aliases:     []string{"flash-attn", "fa"},
				Usage: "Specify enabling Flash Attention, " +
					"which is used to estimate the usage. " +
					"Flash Attention can reduce the usage of RAM/VRAM.",
			},
			&cli.StringFlag{
				Destination: &splitMode,
				Value:       splitMode,
				Category:    "Estimate",
				Name:        "split-mode",
				Aliases:     []string{"sm"},
				Usage: "Specify how to split the model across multiple devices, " +
					"which is used to estimate the usage, select from [layer, row, none]. " +
					"Since gguf-parser always estimates the usage of VRAM, " +
					"\"none\" is meaningless here, keep for compatibility.",
			},
			&cli.StringFlag{
				Destination: &tensorSplit,
				Value:       tensorSplit,
				Category:    "Estimate",
				Name:        "tensor-split",
				Aliases:     []string{"ts"},
				Usage: "Specify the fraction of the model to offload to each device, " +
					"which is used to estimate the usage, " +
					"it is a comma-separated list of integer. " +
					"Since gguf-parser cannot recognize the host GPU devices or RPC servers, " +
					"must explicitly set --tensor-split to indicate how many devices are used.",
			},
			&cli.UintFlag{
				Destination: &mainGPU,
				Value:       mainGPU,
				Category:    "Estimate",
				Name:        "main-gpu",
				Aliases:     []string{"mg"},
				Usage: "Specify the GPU to use for the model (with --split-mode = none) " +
					"or for intermediate results and KV (with --split-mode = row), " +
					"which is used to estimate the usage. " +
					"Since gguf-parser cannot recognize the host GPU devices or RPC servers, " +
					"--main-gpu only works when --tensor-split is set.",
			},
			&cli.StringFlag{
				Destination: &platformFootprint,
				Value:       platformFootprint,
				Category:    "Estimate",
				Name:        "platform-footprint",
				Usage: "Specify the platform footprint(RAM,VRAM) of running host in MiB, " +
					"which is used to estimate the NonUMA usage, " +
					"default is 150,250. " +
					"Different platform always gets different RAM and VRAM footprints, " +
					"for example, within CUDA, 'cudaMemGetInfo' would occupy some RAM and VRAM, " +
					"see https://stackoverflow.com/questions/64854862/free-memory-occupied-by-cudamemgetinfo.",
			},
			&cli.BoolFlag{
				Destination: &noMMap,
				Value:       noMMap,
				Category:    "Estimate",
				Name:        "no-mmap",
				Usage: "Specify disabling Memory-Mapped using, " +
					"which is used to estimate the usage. " +
					"Memory-Mapped can avoid loading the entire model weights into RAM.",
			},
			&cli.IntFlag{
				Destination: &offloadLayers,
				Value:       offloadLayers,
				Category:    "Estimate",
				Name:        "gpu-layers",
				Aliases:     []string{"ngl"},
				Usage: "Specify how many layers of the main model to offload, " +
					"which is used to estimate the usage, " +
					"default is full offloaded.",
			},
			&cli.IntFlag{
				Destination: &offloadLayersDraft,
				Value:       offloadLayersDraft,
				Category:    "Estimate",
				Name:        "gpu-layers-draft",
				Aliases:     []string{"ngld"},
				Usage: "Specify how many layers of the draft model to offload, " +
					"which is used to estimate the usage, " +
					"default is full offloaded.",
			},
			&cli.Uint64Flag{
				Destination: &offloadLayersStep,
				Value:       offloadLayersStep,
				Category:    "Estimate",
				Name:        "gpu-layers-step",
				Usage: "Specify the step of layers to offload, " +
					"works with --gpu-layers.",
			},
			&cli.BoolFlag{
				Destination: &raw,
				Value:       raw,
				Category:    "Output",
				Name:        "raw",
				Usage:       "Output the GGUF file information as JSON only, skip anything.",
			},
			&cli.StringFlag{
				Destination: &rawOutput,
				Value:       rawOutput,
				Category:    "Output",
				Name:        "raw-output",
				Usage:       "Works with --raw, to save the result to the file",
			},
			&cli.BoolFlag{
				Destination: &skipModel,
				Value:       skipModel,
				Category:    "Output",
				Name:        "skip-model",
				Usage:       "Skip to display model metadata.",
			},
			&cli.BoolFlag{
				Destination: &skipArchitecture,
				Value:       skipArchitecture,
				Category:    "Output",
				Name:        "skip-architecture",
				Usage:       "Skip to display architecture metadata.",
			},
			&cli.BoolFlag{
				Destination: &skipTokenizer,
				Value:       skipTokenizer,
				Category:    "Output",
				Name:        "skip-tokenizer",
				Usage:       "Skip to display tokenizer metadata.",
			},
			&cli.BoolFlag{
				Destination: &skipEstimate,
				Value:       skipEstimate,
				Category:    "Output",
				Name:        "skip-estimate",
				Usage:       "Skip to estimate.",
			},
			&cli.BoolFlag{
				Destination: &inMib,
				Value:       inMib,
				Category:    "Output",
				Name:        "in-mib",
				Usage:       "Display the estimated result in table with MiB.",
			},
			&cli.BoolFlag{
				Destination: &inJson,
				Value:       inJson,
				Category:    "Output",
				Name:        "json",
				Usage:       "Output as JSON.",
			},
			&cli.BoolFlag{
				Destination: &inPrettyJson,
				Value:       inPrettyJson,
				Category:    "Output",
				Name:        "json-pretty",
				Usage:       "Works with --json, to output pretty format JSON.",
			},
		},
		Action: mainAction,
	}

	if err := app.RunContext(signalx.Handler(), os.Args); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

var (
	// model options
	path         string
	mmprojPath   string // for estimate
	draftPath    string // for estimate
	url          string
	mmprojUrl    string // for estimate
	draftUrl     string // for estimate
	token        string
	hfRepo       string
	hfFile       string
	hfMMProjFile string // for estimate
	hfDraftRepo  string // for estimate
	hfDraftFile  string // for estimate
	hfToken      string
	msRepo       string
	msFile       string
	msMMProjFile string // for estimate
	msDraftRepo  string // for estimate
	msDraftFile  string // for estimate
	msToken      string
	olModel      string
	olUsage      bool
	// load options
	debug                  bool
	skipProxy              bool
	skipTLSVerify          bool
	skipDNSCache           bool
	skipRangDownloadDetect bool
	skipCache              bool
	// estimate options
	ctxSize            = -1
	inMaxCtxSize       bool
	logicalBatchSize   = 2048
	physicalBatchSize  = 512
	parallelSize       = 1
	cacheKeyType       = "f16"
	cacheValueType     = "f16"
	noKVOffload        bool
	flashAttention     bool
	splitMode          = "layer"
	tensorSplit        string
	mainGPU            uint
	platformFootprint  = "150,250"
	noMMap             bool
	offloadLayers      = -1
	offloadLayersDraft = -1
	offloadLayersStep  uint64
	// output options
	raw              bool
	rawOutput        string
	skipModel        bool
	skipArchitecture bool
	skipTokenizer    bool
	skipEstimate     bool
	inMib            bool
	inJson           bool
	inPrettyJson     = true
)

func mainAction(c *cli.Context) error {
	ctx := c.Context

	// Prepare options.

	ropts := []GGUFReadOption{
		SkipLargeMetadata(),
		UseMMap(),
		UseCache(),
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
	if skipRangDownloadDetect {
		ropts = append(ropts, SkipRangeDownloadDetection())
	}
	if skipCache {
		ropts = append(ropts, SkipCache())
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
	if logicalBatchSize > 0 {
		eopts = append(eopts, WithLogicalBatchSize(int32(max(32, logicalBatchSize))))
	}
	if physicalBatchSize > 0 {
		if physicalBatchSize > logicalBatchSize {
			return errors.New("--ubatch-size must be less than or equal to --batch-size")
		}
		eopts = append(eopts, WithPhysicalBatchSize(int32(physicalBatchSize)))
	}
	if parallelSize > 0 {
		eopts = append(eopts, WithParallelSize(int32(parallelSize)))
	}
	if cacheKeyType != "" {
		eopts = append(eopts, WithCacheKeyType(toGGMLType(cacheKeyType)))
	}
	if cacheValueType != "" {
		eopts = append(eopts, WithCacheValueType(toGGMLType(cacheValueType)))
	}
	if noKVOffload {
		eopts = append(eopts, WithoutOffloadKVCache())
	}
	if flashAttention {
		eopts = append(eopts, WithFlashAttention())
	}
	switch splitMode {
	case "row":
		eopts = append(eopts, WithSplitMode(LLaMACppSplitModeRow))
	case "none":
		eopts = append(eopts, WithSplitMode(LLaMACppSplitModeNone))
	default:
		eopts = append(eopts, WithSplitMode(LLaMACppSplitModeLayer))
	}
	if tensorSplit != "" {
		ss := strings.Split(tensorSplit, ",")
		var vs float64
		vv := make([]float64, len(ss))
		vf := make([]float64, len(ss))
		for i, s := range ss {
			v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
			if err != nil {
				return errors.New("--tensor-split has invalid integer")
			}
			vs += v
			vv[i] = vs
		}
		for i, v := range vv {
			vf[i] = v / vs
		}
		eopts = append(eopts, WithTensorSplitFraction(vf))
		if mainGPU < uint(len(vv)) {
			eopts = append(eopts, WithMainGPUIndex(int(mainGPU)))
		} else {
			return errors.New("--main-gpu must be less than item size of --tensor-split")
		}
	}

	// Parse GGUF file.

	var gf, mmpgf, dftgf *GGUFFile
	{
		var err error

		ropts := ropts[:len(ropts):len(ropts)]

		// Main model.
		switch {
		default:
			return errors.New("no model specified")
		case path != "":
			gf, err = ParseGGUFFile(path, ropts...)
		case url != "":
			gf, err = ParseGGUFFileRemote(ctx, url, ropts...)
		case hfRepo != "" && hfFile != "":
			if hfToken != "" {
				ropts = append(ropts, UseBearerAuth(hfToken))
			}
			gf, err = ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfFile, ropts...)
		case msRepo != "" && msFile != "":
			if msToken != "" {
				ropts = append(ropts, UseBearerAuth(msToken))
			}
			gf, err = ParseGGUFFileFromModelScope(ctx, msRepo, msFile, ropts...)
		case olModel != "":
			om := ParseOllamaModel(olModel)
			gf, err = ParseGGUFFileFromOllamaModel(ctx, om, ropts...)
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
				// Multimodal projector overlap.
				{
					mls := om.SearchLayers(regexp.MustCompile(`^application/vnd\.ollama\.image\.projector$`))
					if len(mls) > 0 {
						mmpgf, err = ParseGGUFFileRemote(ctx, mls[len(mls)-1].BlobURL().String(), ropts...)
					}
				}
			}
		}
		if err != nil {
			return fmt.Errorf("failed to parse GGUF file: %w", err)
		}

		// MultimodalProjector model.
		switch {
		case mmprojPath != "":
			mmpgf, err = ParseGGUFFile(mmprojPath, ropts...)
		case mmprojUrl != "":
			mmpgf, err = ParseGGUFFileRemote(ctx, mmprojUrl, ropts...)
		case hfRepo != "" && hfMMProjFile != "":
			mmpgf, err = ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfMMProjFile, ropts...)
		case msRepo != "" && msMMProjFile != "":
			mmpgf, err = ParseGGUFFileFromModelScope(ctx, msRepo, msMMProjFile, ropts...)
		}
		if err != nil {
			return fmt.Errorf("failed to parse multimodal projector GGUF file: %w", err)
		}

		// Drafter model.
		switch {
		case draftPath != "":
			dftgf, err = ParseGGUFFile(draftPath, ropts...)
		case draftUrl != "":
			dftgf, err = ParseGGUFFileRemote(ctx, draftUrl, ropts...)
		case hfDraftRepo != "" && hfDraftFile != "":
			dftgf, err = ParseGGUFFileFromHuggingFace(ctx, hfDraftRepo, hfDraftFile, ropts...)
		case msDraftRepo != "" && msDraftFile != "":
			dftgf, err = ParseGGUFFileFromModelScope(ctx, msDraftRepo, msDraftFile, ropts...)
		}
		if err != nil {
			return fmt.Errorf("failed to parse draft GGUF file: %w", err)
		}
	}

	// Output raw.

	if raw {
		w := os.Stdout
		if rawOutput != "" {
			f, err := osx.CreateFile(rawOutput, 0o666)
			if err != nil {
				return fmt.Errorf("failed to create file: %w", err)
			}
			defer osx.Close(f)
			w = f
		}
		if err := json.NewEncoder(w).Encode(gf); err != nil {
			return fmt.Errorf("failed to encode JSON: %w", err)
		}
		return nil
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
		if mmpgf != nil {
			meopts := eopts[:len(eopts):len(eopts)]
			me := mmpgf.EstimateLLaMACppUsage(meopts...)
			eopts = append(eopts, WithMultimodalProjector(&me))
		}

		if dftgf != nil {
			deopts := eopts[:len(eopts):len(eopts)]
			if offloadLayersDraft >= 0 {
				deopts = append(deopts, WithOffloadLayers(uint64(offloadLayersDraft)))
			}
			de := dftgf.EstimateLLaMACppUsage(deopts...)
			eopts = append(eopts, WithDrafter(&de))
		}

		deopts := eopts[:len(eopts):len(eopts)]
		if offloadLayers >= 0 {
			deopts = append(deopts, WithOffloadLayers(uint64(offloadLayers)))
		}
		e = gf.EstimateLLaMACppUsage(deopts...)
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
			return fmt.Errorf("failed to encode JSON: %w", err)
		}

		return nil
	}

	InMiBytes = inMib

	if !skipModel {
		tprint(
			"MODEL",
			[][]any{
				{
					"Name",
					"Arch",
					"Quantization",
					"Little Endian",
					"Size",
					"Parameters",
					"BPW",
				},
			},
			[][]any{
				{
					m.Name,
					m.Architecture,
					sprintf(m.FileType),
					sprintf(m.LittleEndian),
					sprintf(m.Size),
					sprintf(m.Parameters),
					sprintf(m.BitsPerWeight),
				},
			})
	}

	if !skipArchitecture {
		var (
			hd []any
			bd []any
		)
		if a.Architecture != "clip" {
			hd = []any{
				"Max Context Len",
				"Embedding Len",
				"Embedding GQA",
				"Attention Causal",
				"Attention Head Cnt",
				"Layers",
				"Feed Forward Len",
				"Expert Cnt",
				"Vocabulary Len",
			}
			bd = []any{
				sprintf(a.MaximumContextLength),
				sprintf(a.EmbeddingLength),
				sprintf(a.EmbeddingGQA),
				sprintf(a.AttentionCausal),
				sprintf(tenary(a.AttentionHeadCountKV == 0 || a.AttentionHeadCountKV == a.AttentionHeadCount, "N/A", a.AttentionHeadCount)),
				sprintf(a.BlockCount),
				sprintf(a.FeedForwardLength),
				sprintf(a.ExpertCount),
				sprintf(a.VocabularyLength),
			}
		} else {
			hd = []any{
				"Embedding Len",
				"Layers",
				"Feed Forward Len",
				"Encoder",
				"LLaVA MultimodalProjector",
			}
			bd = []any{
				sprintf(a.EmbeddingLength),
				sprintf(a.BlockCount),
				sprintf(a.FeedForwardLength),
				sprintf(tenary(a.ClipHasTextEncoder, tenary(a.ClipHasVisionEncoder, "Text & Vision", "Text"), tenary(a.ClipHasVisionEncoder, "Vision", "N/A"))),
				sprintf(tenary(a.ClipHasLLaVaProjector, a.ClipProjectorType, "N/A")),
			}
		}
		tprint(
			"ARCHITECTURE",
			[][]any{hd},
			[][]any{bd})
	}

	if !skipTokenizer && t.Model != "" {
		tprint(
			"TOKENIZER",
			[][]any{
				{
					"Model",
					"Tokens Size",
					"Tokens Len",
					"Added Tokens Len",
					"BOS Token",
					"EOS Token",
					"EOT Token",
					"EOM Token",
					"Unknown Token",
					"Separator Token",
					"Padding Token",
				},
			},
			[][]any{
				{
					t.Model,
					sprintf(tenary(t.TokensSize <= 0, "N/A", GGUFBytesScalar(t.TokensSize))),
					sprintf(tenary(t.TokensLength <= 0, "N/A", t.TokensLength)),
					sprintf(tenary(t.AddedTokensLength <= 0, "N/A", t.AddedTokensLength)),
					sprintf(tenary(t.BOSTokenID < 0, "N/A", t.BOSTokenID)),
					sprintf(tenary(t.EOSTokenID < 0, "N/A", t.EOSTokenID)),
					sprintf(tenary(t.EOTTokenID < 0, "N/A", t.EOTTokenID)),
					sprintf(tenary(t.EOMTokenID < 0, "N/A", t.EOMTokenID)),
					sprintf(tenary(t.UnknownTokenID < 0, "N/A", t.UnknownTokenID)),
					sprintf(tenary(t.SeparatorTokenID < 0, "N/A", t.SeparatorTokenID)),
					sprintf(tenary(t.PaddingTokenID < 0, "N/A", t.PaddingTokenID)),
				},
			})
	}

	if !skipEstimate {
		var (
			hds [][]any
			bds [][]any
		)
		es := e.Summarize(mmap, platformRAM, platformVRAM)
		if e.Architecture != "clip" {
			hds = [][]any{
				{
					"Arch",
					"Context Size",
					"Batch Size (L / P)",
					"Flash Attention",
					"MMap Load",
					"Embedding Only",
					"Offload Layers",
					"Full Offloaded",
					"RAM",
					"RAM",
				},
				{
					"Arch",
					"Context Size",
					"Batch Size (L / P)",
					"Flash Attention",
					"MMap Load",
					"Embedding Only",
					"Offload Layers",
					"Full Offloaded",
					"UMA",
					"NonUMA",
				},
			}
			for i := range es.Memory[0].VRAMs {
				hds[0] = append(hds[0], fmt.Sprintf("VRAM %d", i), fmt.Sprintf("VRAM %d", i))
				hds[1] = append(hds[1], "UMA", "NonUMA")
			}

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

			bds = make([][]any, len(es.Memory))
			for i := range es.Memory {
				bds[i] = []any{
					sprintf(es.Architecture),
					sprintf(es.ContextSize),
					sprintf("%d / %d", es.LogicalBatchSize, es.PhysicalBatchSize),
					sprintf(tenary(es.FlashAttention, "Enabled", "Disabled")),
					sprintf(tenary(!es.NoMMap, "Supported", "Not Supported")),
					sprintf(tenary(es.EmbeddingOnly, "Yes", "No")),
					sprintf(tenary(es.Memory[i].FullOffloaded, sprintf("%d (%d + 1)",
						es.Memory[i].OffloadLayers, es.Memory[i].OffloadLayers-1), es.Memory[i].OffloadLayers)),
					sprintf(tenary(es.Memory[i].FullOffloaded, "Yes", "No")),
					sprintf(es.Memory[i].RAM.UMA),
					sprintf(es.Memory[i].RAM.NonUMA),
				}
				for _, v := range es.Memory[i].VRAMs {
					bds[i] = append(bds[i],
						sprintf(v.UMA),
						sprintf(v.NonUMA))
				}
			}
		} else {
			hds = [][]any{
				{
					"Arch",
					"Offload Layers",
					"Full Offloaded",
					"RAM",
					"RAM",
				},
				{
					"Arch",
					"Offload Layers",
					"Full Offloaded",
					"UMA",
					"NonUMA",
				},
			}
			for i := range es.Memory[0].VRAMs {
				hds[0] = append(hds[0], fmt.Sprintf("VRAM %d", i), fmt.Sprintf("VRAM %d", i))
				hds[1] = append(hds[1], "UMA", "NonUMA")
			}

			bds = [][]any{
				{
					sprintf(es.Architecture),
					sprintf(es.Memory[0].OffloadLayers),
					sprintf(tenary(es.Memory[0].FullOffloaded, "Yes", "No")),
					sprintf(es.Memory[0].RAM.UMA),
					sprintf(es.Memory[0].RAM.NonUMA),
				},
			}
			for _, v := range es.Memory[0].VRAMs {
				bds[0] = append(bds[0],
					sprintf(v.UMA),
					sprintf(v.NonUMA))
			}
		}
		tprint(
			"ESTIMATE",
			hds,
			bds)
	}

	return nil
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

func tprint(title string, headers, bodies [][]any) {
	tw := table.NewWriter()
	tw.SetOutputMirror(os.Stdout)
	tw.SetTitle(strings.ToUpper(title))
	for i := range headers {
		tw.AppendHeader(headers[i], table.RowConfig{AutoMerge: true, AutoMergeAlign: text.AlignCenter})
	}
	for i := range bodies {
		tw.AppendRow(bodies[i])
	}
	tw.SetColumnConfigs(func() (r []table.ColumnConfig) {
		r = make([]table.ColumnConfig, len(headers[0]))
		for i := range r {
			r[i].Number = i + 1
			r[i].AutoMerge = true
			r[i].Align = text.AlignCenter
			r[i].AlignHeader = text.AlignCenter
		}
		return r
	}())
	tw.Style().Options.SeparateRows = true
	tw.Render()
	fmt.Println()
}

func tenary(c bool, t, f any) any {
	if c {
		return t
	}
	return f
}

func toGGMLType(s string) GGMLType {
	t := GGMLTypeF16
	switch s {
	case "f32":
		t = GGMLTypeF32
	case "f16":
		t = GGMLTypeF16
	case "q8_0":
		t = GGMLTypeQ8_0
	case "q4_0":
		t = GGMLTypeQ4_0
	case "q4_1":
		t = GGMLTypeQ4_1
	case "iq4_nl":
		t = GGMLTypeIQ4_NL
	case "q5_0":
		t = GGMLTypeQ5_0
	case "q5_1":
		t = GGMLTypeQ5_1
	}
	return t
}
