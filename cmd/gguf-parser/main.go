package main

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

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
		Usage:           "Review/Check GGUF files and estimate the memory usage.",
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
			&cli.StringSliceFlag{
				Destination: &loraPaths,
				Category:    "Model/Local",
				Name:        "lora-path",
				Aliases:     []string{"lora"},
				Usage:       "Path where the GGUF file to load for the LoRA adapter, optional.",
			},
			&cli.StringSliceFlag{
				Destination: &controlVectorPaths,
				Category:    "Model/Local",
				Name:        "control-vector-path",
				Aliases:     []string{"control-vector"},
				Usage:       "Path where the GGUF file to load for the Control Vector adapter, optional.",
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
			&cli.StringSliceFlag{
				Destination: &loraUrls,
				Category:    "Model/Remote",
				Name:        "lora-url",
				Usage:       "Url where the GGUF file to load for the LoRA adapter, optional.",
			},
			&cli.StringSliceFlag{
				Destination: &controlVectorUrls,
				Category:    "Model/Remote",
				Name:        "control-vector-url",
				Usage:       "Url where the GGUF file to load for the Control Vector adapter, optional.",
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
				Destination: &hfMMProjFile,
				Value:       hfMMProjFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-mmproj-file",
				Usage:       "Multimodal projector file below the --hf-repo.",
			},
			&cli.StringSliceFlag{
				Destination: &hfLoRAFiles,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-lora-file",
				Usage:       "LoRA adapter file below the --hf-repo.",
			},
			&cli.StringSliceFlag{
				Destination: &hfControlVectorFiles,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-control-vector-file",
				Usage:       "Control Vector adapter file below the --hf-repo.",
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
				Destination: &msMMProjFile,
				Value:       msMMProjFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-mmproj-file",
				Usage:       "Multimodal projector file below the --ms-repo.",
			},
			&cli.StringSliceFlag{
				Destination: &msLoRAFiles,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-lora-file",
				Usage:       "LoRA adapter file below the --ms-repo.",
			},
			&cli.StringSliceFlag{
				Destination: &msControlVectorFiles,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-control-vector-file",
				Usage:       "Control Vector adapter file below the --ms-repo.",
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
				Destination: &olBaseURL,
				Value:       olBaseURL,
				Category:    "Model/Remote/Ollama",
				Name:        "ol-base-url",
				Usage: "Model base URL of Ollama, e.g. " +
					"https://registry.ollama.ai.",
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
			&cli.DurationFlag{
				Destination: &cacheExpiration,
				Value:       cacheExpiration,
				Category:    "Load",
				Name:        "cache-expiration",
				Usage: "Specify the expiration of cache, " +
					"works with --url/--hf-*/--ms-*/--ol-*.",
			},
			&cli.StringFlag{
				Destination: &cachePath,
				Value:       cachePath,
				Category:    "Load",
				Name:        "cache-path",
				Usage: "Cache the read result to the path, " +
					"works with --url/--hf-*/--ms-*/--ol-*.",
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
					"must explicitly set --tensor-split to indicate how many devices are used. " +
					"To declare the devices belong to RPC servers, set --rpc please.",
			},
			&cli.StringFlag{
				Destination: &rpcServers,
				Value:       rpcServers,
				Category:    "Estimate",
				Name:        "rpc",
				Usage: "Specify the RPC servers, " +
					"which is used to estimate the usage, " +
					"it is a comma-separated list of host:port. " +
					"Woks with --tensor-split.",
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
				Aliases:     []string{"ngl", "n-gpu-layers"},
				Usage: "Specify how many layers of the main model to offload, " +
					"which is used to estimate the usage, " +
					"default is full offloaded.",
			},
			&cli.IntFlag{
				Destination: &offloadLayersDraft,
				Value:       offloadLayersDraft,
				Category:    "Estimate",
				Name:        "gpu-layers-draft",
				Aliases:     []string{"ngld", "n-gpu-layers-draft"},
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
				Destination: &skipMetadata,
				Value:       skipMetadata,
				Category:    "Output",
				Name:        "skip-metadata",
				Usage:       "Skip to display metadata.",
			},
			&cli.BoolFlag{
				Destination: &skipArchitecture,
				Value:       skipArchitecture,
				Category:    "Output",
				Name:        "skip-architecture",
				Usage:       "Skip to display architecture.",
			},
			&cli.BoolFlag{
				Destination: &skipTokenizer,
				Value:       skipTokenizer,
				Category:    "Output",
				Name:        "skip-tokenizer",
				Usage: "Skip to display tokenizer. " +
					"By default, gguf-parser always displays the tokenizer of the file which types with \"model\".",
			},
			&cli.BoolFlag{
				Destination: &skipEstimate,
				Value:       skipEstimate,
				Category:    "Output",
				Name:        "skip-estimate",
				Usage: "Skip to estimate. " +
					"By default, gguf-parser always estimates the file which types with \"model\".",
			},
			&cli.BoolFlag{
				Destination: &inShort,
				Value:       inShort,
				Category:    "Output",
				Name:        "in-short",
				Usage:       "Display the estimated result in table in short form.",
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
	path                 string
	mmprojPath           string          // for estimate
	draftPath            string          // for estimate
	loraPaths            cli.StringSlice // for estimate
	controlVectorPaths   cli.StringSlice // for estimate
	url                  string
	mmprojUrl            string          // for estimate
	draftUrl             string          // for estimate
	loraUrls             cli.StringSlice // for estimate
	controlVectorUrls    cli.StringSlice // for estimate
	token                string
	hfRepo               string
	hfFile               string
	hfDraftRepo          string          // for estimate
	hfDraftFile          string          // for estimate
	hfMMProjFile         string          // for estimate
	hfLoRAFiles          cli.StringSlice // for estimate
	hfControlVectorFiles cli.StringSlice // for estimate
	hfToken              string
	msRepo               string
	msFile               string
	msDraftRepo          string          // for estimate
	msDraftFile          string          // for estimate
	msMMProjFile         string          // for estimate
	msLoRAFiles          cli.StringSlice // for estimate
	msControlVectorFiles cli.StringSlice // for estimate
	msToken              string
	olBaseURL            = "https://registry.ollama.ai"
	olModel              string
	olUsage              bool
	// load options
	debug                  bool
	skipProxy              bool
	skipTLSVerify          bool
	skipDNSCache           bool
	skipRangDownloadDetect bool
	cacheExpiration        = 24 * time.Hour
	cachePath              = DefaultCachePath()
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
	rpcServers         string
	platformFootprint  = "150,250"
	noMMap             bool
	offloadLayers      = -1
	offloadLayersDraft = -1
	offloadLayersStep  uint64
	// output options
	raw              bool
	rawOutput        string
	inShort          bool
	skipMetadata     bool
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
	if cacheExpiration > 0 {
		ropts = append(ropts, UseCacheExpiration(cacheExpiration))
	}
	if cachePath != "" {
		ropts = append(ropts, UseCachePath(cachePath))
	}
	if skipCache {
		ropts = append(ropts, SkipCache())
	}

	eopts := []LLaMACppRunEstimateOption{
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
		tss := strings.Split(tensorSplit, ",")
		var vs float64
		vv := make([]float64, len(tss))
		vf := make([]float64, len(tss))
		for i, s := range tss {
			s = strings.TrimSpace(s)
			v, err := strconv.ParseFloat(s, 64)
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
		if rpcServers != "" {
			rss := strings.Split(rpcServers, ",")
			if len(rss) > len(tss) {
				return errors.New("--rpc has more items than --tensor-split")
			}
			rpc := make([]string, len(rss))
			for i, s := range rss {
				s = strings.TrimSpace(s)
				if _, _, err := net.SplitHostPort(s); err != nil {
					return errors.New("--rpc has invalid host:port")
				}
				rpc[i] = s
			}
			eopts = append(eopts, WithRPCServers(rpc))
		}
	}

	// Parse GGUF file.

	var (
		gf     *GGUFFile
		projgf *GGUFFile
		dftgf  *GGUFFile
		adpgfs []*GGUFFile
	)
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
			om := ParseOllamaModel(olModel, SetOllamaModelBaseURL(olBaseURL))
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
						projgf, err = ParseGGUFFileRemote(ctx, mls[len(mls)-1].BlobURL().String(), ropts...)
					}
				}
				// Adapter overlap.
				{
					als := om.SearchLayers(regexp.MustCompile(`^application/vnd\.ollama\.image\.adapter$`))
					if len(als) > 0 {
						var adpgf *GGUFFile
						for i := range als {
							adpgf, err = ParseGGUFFileRemote(ctx, als[i].BlobURL().String(), ropts...)
							if err != nil {
								break
							}
							adpgfs = append(adpgfs, adpgf)
						}
					}
				}
			}
		}
		if err != nil {
			return fmt.Errorf("failed to parse GGUF file: %w", err)
		}

		// Drafter.
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

		// Projector.
		switch {
		case mmprojPath != "":
			projgf, err = ParseGGUFFile(mmprojPath, ropts...)
		case mmprojUrl != "":
			projgf, err = ParseGGUFFileRemote(ctx, mmprojUrl, ropts...)
		case hfRepo != "" && hfMMProjFile != "":
			projgf, err = ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfMMProjFile, ropts...)
		case msRepo != "" && msMMProjFile != "":
			projgf, err = ParseGGUFFileFromModelScope(ctx, msRepo, msMMProjFile, ropts...)
		}
		if err != nil {
			return fmt.Errorf("failed to parse multimodal projector GGUF file: %w", err)
		}

		// Adapter.
		{
			// LoRA.
			for _, loraPath := range loraPaths.Value() {
				adpgf, err := ParseGGUFFile(loraPath, ropts...)
				if err != nil {
					return fmt.Errorf("failed to parse LoRA adapter GGUF file: %w", err)
				}
				adpgfs = append(adpgfs, adpgf)
			}
			for _, loraUrl := range loraUrls.Value() {
				adpgf, err := ParseGGUFFileRemote(ctx, loraUrl, ropts...)
				if err != nil {
					return fmt.Errorf("failed to parse LoRA adapter GGUF file: %w", err)
				}
				adpgfs = append(adpgfs, adpgf)
			}
			if hfRepo != "" {
				for _, hfLoRAFile := range hfLoRAFiles.Value() {
					adpgf, err := ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfLoRAFile, ropts...)
					if err != nil {
						return fmt.Errorf("failed to parse LoRA adapter GGUF file: %w", err)
					}
					adpgfs = append(adpgfs, adpgf)
				}
			}
			if msRepo != "" {
				for _, msLoRAFile := range msLoRAFiles.Value() {
					adpgf, err := ParseGGUFFileFromModelScope(ctx, msRepo, msLoRAFile, ropts...)
					if err != nil {
						return fmt.Errorf("failed to parse LoRA adapter GGUF file: %w", err)
					}
					adpgfs = append(adpgfs, adpgf)
				}
			}

			// Control Vector.
			for _, cvPath := range controlVectorPaths.Value() {
				adpgf, err := ParseGGUFFile(cvPath, ropts...)
				if err != nil {
					return fmt.Errorf("failed to parse Control Vector adapter GGUF file: %w", err)
				}
				adpgfs = append(adpgfs, adpgf)
			}
			for _, cvUrl := range controlVectorUrls.Value() {
				adpgf, err := ParseGGUFFileRemote(ctx, cvUrl, ropts...)
				if err != nil {
					return fmt.Errorf("failed to parse Control Vector adapter GGUF file: %w", err)
				}
				adpgfs = append(adpgfs, adpgf)
			}
			if hfRepo != "" {
				for _, hfCvFile := range hfControlVectorFiles.Value() {
					adpgf, err := ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfCvFile, ropts...)
					if err != nil {
						return fmt.Errorf("failed to parse Control Vector adapter GGUF file: %w", err)
					}
					adpgfs = append(adpgfs, adpgf)
				}
			}
			if msRepo != "" {
				for _, msCvFile := range msControlVectorFiles.Value() {
					adpgf, err := ParseGGUFFileFromModelScope(ctx, msRepo, msCvFile, ropts...)
					if err != nil {
						return fmt.Errorf("failed to parse Control Vector adapter GGUF file: %w", err)
					}
					adpgfs = append(adpgfs, adpgf)
				}
			}
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
		m GGUFMetadata
		a GGUFArchitecture
		t GGUFTokenizer
		e LLaMACppRunEstimate
	)
	if !skipMetadata {
		m = gf.Metadata()
	}
	if !skipArchitecture {
		a = gf.Architecture()
	}
	if !skipTokenizer {
		t = gf.Tokenizer()
	}
	if !skipEstimate {
		if dftgf != nil {
			deopts := eopts[:len(eopts):len(eopts)]
			if offloadLayersDraft >= 0 {
				deopts = append(deopts, WithOffloadLayers(uint64(offloadLayersDraft)))
			}
			de := dftgf.EstimateLLaMACppRun(deopts...)
			eopts = append(eopts, WithDrafter(&de))
		}

		if projgf != nil {
			peopts := eopts[:len(eopts):len(eopts)]
			me := projgf.EstimateLLaMACppRun(peopts...)
			eopts = append(eopts, WithProjector(&me))
		}

		if len(adpgfs) > 0 {
			adps := make([]LLaMACppRunEstimate, len(adpgfs))
			aeopts := eopts[:len(eopts):len(eopts)]
			for i, adpgf := range adpgfs {
				ae := adpgf.EstimateLLaMACppRun(aeopts...)
				adps[i] = ae
			}
			eopts = append(eopts, WithAdapters(adps))
		}

		deopts := eopts[:len(eopts):len(eopts)]
		if offloadLayers >= 0 {
			deopts = append(deopts, WithOffloadLayers(uint64(offloadLayers)))
		}
		e = gf.EstimateLLaMACppRun(deopts...)
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
		if !skipMetadata {
			o["metadata"] = m
		}
		if !skipArchitecture {
			o["architecture"] = a
		}
		if !skipTokenizer && t.Model != "" {
			o["tokenizer"] = t
		}
		if !skipEstimate && e.Type == "model" {
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
						ess[i] = gf.EstimateLLaMACppRun(eopts...).SummarizeMemory(mmap, platformRAM, platformVRAM)
					}(i)
				}
				wg.Wait()
				ess[cap(ess)-1] = es.Memory[0]
				es.Memory = ess
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

	if !skipMetadata {
		tprint(
			"Metadata",
			[][]any{
				{
					"Type",
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
					m.Type,
					sprintf(tenary(len(m.Name) == 0, "N/A", tenary(len([]rune(m.Name)) <= 20, m.Name, string([]rune(m.Name)[:20])+"..."))),
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
		switch a.Type {
		case "projector":
			hd = []any{
				"Projector Type",
				"Embedding Len",
				"Layers",
				"Feed Forward Len",
				"Encoder",
			}
			bd = []any{
				sprintf(a.ClipProjectorType),
				sprintf(a.EmbeddingLength),
				sprintf(a.BlockCount),
				sprintf(a.FeedForwardLength),
				sprintf(tenary(a.ClipHasTextEncoder, tenary(a.ClipHasVisionEncoder, "Text & Vision", "Text"), tenary(a.ClipHasVisionEncoder, "Vision", "N/A"))),
			}
		case "adapter":
			hd = []any{
				"Adapter Type",
			}
			bd = []any{
				sprintf(a.AdapterType),
			}
			if a.AdapterType == "lora" {
				hd = append(hd, "LoRA Alpha")
				bd = append(bd, sprintf(a.AdapterLoRAAlpha))
			} else {
				hd = append(hd, "ControlVector Layers")
				bd = append(bd, sprintf(a.AdapterControlVectorLayerCount))
			}
		default:
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

	if !skipEstimate && e.Type == "model" {
		hds := make([][]any, 2)
		es := e.Summarize(mmap, platformRAM, platformVRAM)
		if !inShort {
			hds[0] = []any{
				"Arch",
				"Context Size",
				"Batch Size (L / P)",
				"Flash Attention",
				"MMap Load",
				"Embedding Only",
				"Distributable",
			}
			hds[1] = []any{
				"Arch",
				"Context Size",
				"Batch Size (L / P)",
				"Flash Attention",
				"MMap Load",
				"Embedding Only",
				"Distributable",
			}
		}
		hds[0] = append(hds[0], "Offload Layers", "Full Offloaded", "RAM", "RAM", "RAM")
		hds[1] = append(hds[1], "Offload Layers", "Full Offloaded", "Layers", "UMA", "NonUMA")
		for i := range es.Memory[0].VRAMs {
			hds[0] = append(hds[0], fmt.Sprintf("VRAM %d", i), fmt.Sprintf("VRAM %d", i), fmt.Sprintf("VRAM %d", i))
			hds[1] = append(hds[1], "Layers", "UMA", "NonUMA")
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
					ess[i] = gf.EstimateLLaMACppRun(eopts...).SummarizeMemory(mmap, platformRAM, platformVRAM)
				}(i)
			}
			wg.Wait()
			ess[cap(ess)-1] = es.Memory[0]
			es.Memory = ess
		}

		bds := make([][]any, len(es.Memory))
		for i := range es.Memory {
			if !inShort {
				bds[i] = []any{
					sprintf(es.Architecture),
					sprintf(es.ContextSize),
					sprintf("%d / %d", es.LogicalBatchSize, es.PhysicalBatchSize),
					sprintf(tenary(flashAttention, tenary(es.FlashAttention, "Enabled", "Not Supported"), "Disabled")),
					sprintf(tenary(mmap, tenary(!es.NoMMap, "Enabled", "Not Supported"), "Disabled")),
					sprintf(tenary(es.EmbeddingOnly, "Yes", "No")),
					sprintf(tenary(es.Distributable, "Supported", "Not Supported")),
				}
			}
			bds[i] = append(bds[i],
				sprintf(tenary(es.Memory[i].FullOffloaded, sprintf("%d (%d + 1)",
					es.Memory[i].OffloadLayers, es.Memory[i].OffloadLayers-1), es.Memory[i].OffloadLayers)),
				sprintf(tenary(es.Memory[i].FullOffloaded, "Yes", "No")),
				sprintf(tenary(!es.Memory[i].RAM.HandleOutputLayer, es.Memory[i].RAM.HandleLayers, sprintf("%d + 1", es.Memory[i].RAM.HandleLayers))),
				sprintf(es.Memory[i].RAM.UMA),
				sprintf(es.Memory[i].RAM.NonUMA))
			for _, v := range es.Memory[i].VRAMs {
				bds[i] = append(bds[i],
					sprintf(tenary(!v.HandleOutputLayer, v.HandleLayers, sprintf("%d + 1", v.HandleLayers))),
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
			if len(headers) > 1 && (headers[1][i] == "Layers" || headers[1][i] == "UMA" || headers[1][i] == "NonUMA") {
				r[i].AutoMerge = false
			}
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
