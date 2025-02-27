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
		Usage:           "Review/Check GGUF files and estimate the memory usage and provide optimization suggestions.",
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
				Aliases: []string{ // LLaMACpp compatibility
					"model",
					"m",
				},
				Usage: "Path where the GGUF file to load for the main model, e.g. \"~/.cache" +
					"/lm-studio/models/QuantFactory/Qwen2-7B-Instruct-GGUF" +
					"/Qwen2-7B-Instruct.Q5_K_M.gguf\".",
			},
			&cli.StringFlag{
				Destination: &draftPath,
				Value:       draftPath,
				Category:    "Model/Local",
				Name:        "draft-path",
				Aliases: []string{ // LLaMACpp compatibility
					"model-draft",
					"md",
				},
				Usage: "Path where the GGUF file to load for the draft model, optional, e.g. \"~/.cache" +
					"/lm-studio/models/QuantFactory/Qwen2-1.5B-Instruct-GGUF" +
					"/Qwen2-1.5B-Instruct.Q5_K_M.gguf\".",
			},
			&cli.StringFlag{
				Destination: &mmprojPath,
				Value:       mmprojPath,
				Category:    "Model/Local",
				Name:        "mmproj-path",
				Aliases: []string{ // LLaMACpp compatibility
					"mmproj",
				},
				Usage: "Path where the GGUF file to load for the multimodal projector, optional.",
			},
			&cli.StringSliceFlag{
				Destination: &loraPaths,
				Category:    "Model/Local",
				Name:        "lora-path",
				Aliases: []string{ // LLaMACpp compatibility
					"lora",
				},
				Usage: "Path where the GGUF file to load for the LoRA adapter, optional.",
			},
			&cli.StringSliceFlag{
				Destination: &controlVectorPaths,
				Category:    "Model/Local",
				Name:        "control-vector-path",
				Aliases: []string{ // LLaMACpp compatibility
					"control-vector",
				},
				Usage: "Path where the GGUF file to load for the Control Vector adapter, optional.",
			},
			&cli.StringFlag{
				Destination: &upscalePath,
				Value:       upscalePath,
				Category:    "Model/Local",
				Name:        "upscale-path",
				Aliases: []string{
					"upscale-model",       // StableDiffusionCpp compatibility
					"image-upscale-model", // LLaMABox compatibility
				},
				Usage: "Path where the GGUF file to load for the Upscale model, optional.",
			},
			&cli.StringFlag{
				Destination: &controlNetPath,
				Value:       controlNetPath,
				Category:    "Model/Local",
				Name:        "control-net-path",
				Aliases: []string{
					"control-net",             // StableDiffusionCpp compatibility
					"image-control-net-model", // LLaMABox compatibility
				},
				Usage: "Path where the GGUF file to load for the Control Net model, optional.",
			},
			&cli.StringFlag{
				Destination: &url,
				Value:       url,
				Category:    "Model/Remote",
				Name:        "url",
				Aliases: []string{
					"model-url",
					"mu",
				},
				Usage: "Url where the GGUF file to load for the main model, e.g. " +
					"\"https://huggingface.co/QuantFactory/Qwen2-7B-Instruct-GGUF" +
					"/resolve/main/Qwen2-7B-Instruct.Q5_K_M.gguf\". " +
					"Note that gguf-parser does not need to download the entire GGUF file.",
			},
			&cli.StringFlag{
				Destination: &draftUrl,
				Value:       draftUrl,
				Category:    "Model/Remote",
				Name:        "draft-url",
				Usage: "Url where the GGUF file to load for the draft model, optional, e.g. " +
					"\"https://huggingface.co/QuantFactory/Qwen2-1.5B-Instruct-GGUF" +
					"/resolve/main/Qwen2-1.5B-Instruct.Q5_K_M.gguf\". " +
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
				Destination: &upscaleUrl,
				Value:       upscaleUrl,
				Category:    "Model/Remote",
				Name:        "upscale-url",
				Usage:       "Url where the GGUF file to load for the Upscale model, optional.",
			},
			&cli.StringFlag{
				Destination: &controlNetUrl,
				Value:       controlNetUrl,
				Category:    "Model/Remote",
				Name:        "control-net-url",
				Usage:       "Url where the GGUF file to load for the Control Net model, optional.",
			},
			&cli.StringFlag{
				Destination: &token,
				Value:       token,
				Category:    "Model/Remote",
				Name:        "token",
				Usage: "Bearer auth token to load GGUF file, optional, " +
					"works with \"--url/--draft-url\".",
			},
			&cli.StringFlag{
				Destination: &hfRepo,
				Value:       hfRepo,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-repo",
				Aliases: []string{ // LLaMACpp compatibility
					"hfr",
				},
				Usage: "Repository of HuggingFace which the GGUF file store for the main model, e.g. " +
					"\"QuantFactory/Qwen2-7B-Instruct-GGUF\", works with \"--hf-file\".",
			},
			&cli.StringFlag{
				Destination: &hfFile,
				Value:       hfFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-file",
				Aliases: []string{ // LLaMACpp compatibility
					"hff",
				},
				Usage: "Model file below the \"--hf-repo\", e.g. " +
					"\"Qwen2-7B-Instruct.Q5_K_M.gguf\".",
			},
			&cli.StringFlag{
				Destination: &hfDraftRepo,
				Value:       hfDraftRepo,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-draft-repo",
				Usage: "Repository of HuggingFace which the GGUF file store for the draft model, optional, e.g. " +
					"\"QuantFactory/Qwen2-1.5B-Instruct-GGUF\", works with \"--hf-draft-file\".",
			},
			&cli.StringFlag{
				Destination: &hfDraftFile,
				Value:       hfDraftFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-draft-file",
				Usage: "Model file below the \"--hf-draft-repo\", optional, e.g. " +
					"\"Qwen2-1.5B-Instruct.Q5_K_M.gguf\".",
			},
			&cli.StringFlag{
				Destination: &hfMMProjFile,
				Value:       hfMMProjFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-mmproj-file",
				Usage:       "Multimodal projector file below the \"--hf-repo\".",
			},
			&cli.StringSliceFlag{
				Destination: &hfLoRAFiles,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-lora-file",
				Usage:       "LoRA adapter file below the \"--hf-repo\".",
			},
			&cli.StringSliceFlag{
				Destination: &hfControlVectorFiles,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-control-vector-file",
				Usage:       "Control Vector adapter file below the \"--hf-repo\".",
			},
			&cli.StringFlag{
				Destination: &hfUpscaleRepo,
				Value:       hfUpscaleRepo,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-upscale-repo",
				Usage: "Repository of HuggingFace which the GGUF file store for the Upscale model, optional, " +
					"works with \"--hf-upscale-file\".",
			},
			&cli.StringFlag{
				Destination: &hfUpscaleFile,
				Value:       hfUpscaleFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-upscale-file",
				Usage:       "Model file below the \"--hf-upscale-repo\", optional.",
			},
			&cli.StringFlag{
				Destination: &hfControlNetRepo,
				Value:       hfControlNetRepo,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-control-net-repo",
				Usage: "Repository of HuggingFace which the GGUF file store for the Control Net model, optional, " +
					"works with \"--hf-control-net-file\".",
			},
			&cli.StringFlag{
				Destination: &hfControlNetFile,
				Value:       hfControlNetFile,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-control-net-file",
				Usage:       "Model file below the \"--hf-control-net-repo\", optional.",
			},
			&cli.StringFlag{
				Destination: &hfToken,
				Value:       hfToken,
				Category:    "Model/Remote/HuggingFace",
				Name:        "hf-token",
				Aliases: []string{ // LLaMACpp compatibility
					"hft",
				},
				Usage: "User access token of HuggingFace, optional, " +
					"works with \"--hf-repo/--hf-file pair\" or \"--hf-draft-repo/--hf-draft-file\" pair. " +
					"See https://huggingface.co/settings/tokens.",
			},
			&cli.StringFlag{
				Destination: &msRepo,
				Value:       msRepo,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-repo",
				Usage: "Repository of ModelScope which the GGUF file store for the main model, e.g. " +
					"\"qwen/Qwen1.5-7B-Chat-GGUF\", works with \"--ms-file\".",
			},
			&cli.StringFlag{
				Destination: &msFile,
				Value:       msFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-file",
				Usage: "Model file below the \"--ms-repo\", e.g. " +
					"\"qwen1_5-7b-chat-q5_k_m.gguf\".",
			},
			&cli.StringFlag{
				Destination: &msDraftRepo,
				Value:       msDraftRepo,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-draft-repo",
				Usage: "Repository of ModelScope which the GGUF file store for the draft model, optional, e.g. " +
					"\"qwen/Qwen1.5-1.8B-Chat-GGUF\", works with \"--ms-draft-file\".",
			},
			&cli.StringFlag{
				Destination: &msDraftFile,
				Value:       msDraftFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-draft-file",
				Usage: "Model file below the \"--ms-draft-repo\", optional, e.g. " +
					"\"qwen1_5-1_8b-chat-q5_k_m.gguf\".",
			},
			&cli.StringFlag{
				Destination: &msMMProjFile,
				Value:       msMMProjFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-mmproj-file",
				Usage:       "Multimodal projector file below the \"--ms-repo\".",
			},
			&cli.StringSliceFlag{
				Destination: &msLoRAFiles,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-lora-file",
				Usage:       "LoRA adapter file below the \"--ms-repo\".",
			},
			&cli.StringSliceFlag{
				Destination: &msControlVectorFiles,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-control-vector-file",
				Usage:       "Control Vector adapter file below the \"--ms-repo\".",
			},
			&cli.StringFlag{
				Destination: &msUpscaleRepo,
				Value:       msUpscaleRepo,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-upscale-repo",
				Usage: "Repository of ModelScope which the GGUF file store for the Upscale model, optional, " +
					"works with \"--ms-upscale-file\".",
			},
			&cli.StringFlag{
				Destination: &msUpscaleFile,
				Value:       msUpscaleFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-upscale-file",
				Usage:       "Model file below the \"--ms-upscale-repo\", optional.",
			},
			&cli.StringFlag{
				Destination: &msControlNetRepo,
				Value:       msControlNetRepo,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-control-net-repo",
				Usage: "Repository of ModelScope which the GGUF file store for the Control Net model, optional, " +
					"works with \"--ms-control-net-file\".",
			},
			&cli.StringFlag{
				Destination: &msControlNetFile,
				Value:       msControlNetFile,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-control-net-file",
				Usage:       "Model file below the \"--ms-control-net-repo\", optional.",
			},
			&cli.StringFlag{
				Destination: &msToken,
				Value:       msToken,
				Category:    "Model/Remote/ModelScope",
				Name:        "ms-token",
				Usage: "Git access token of ModelScope, optional, " +
					"works with \"--ms-repo/--ms-file\" pair or \"--ms-draft-repo/--ms-draft-file\" pair. " +
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
					"\"gemma2\".",
			},
			&cli.BoolFlag{
				Destination: &olUsage,
				Value:       olUsage,
				Category:    "Model/Remote/Ollama",
				Name:        "ol-usage",
				Usage: "Specify respecting the extending layers introduced by Ollama, " +
					"works with \"--ol-model\", which affects the usage estimation.",
			},
			&cli.BoolFlag{
				Destination: &skipProxy,
				Value:       skipProxy,
				Category:    "Load",
				Name:        "skip-proxy",
				Usage: "Skip proxy settings, " +
					"works with \"--url/--hf-*/--ms-*/--ol-*\", " +
					"default is respecting the environment variables \"HTTP_PROXY/HTTPS_PROXY/NO_PROXY\".",
			},
			&cli.BoolFlag{
				Destination: &skipTLSVerify,
				Value:       skipTLSVerify,
				Category:    "Load",
				Name:        "skip-tls-verify",
				Usage: "Skip TLS verification, " +
					"works with \"--url/--hf-*/--ms-*/--ol-*\", " +
					"default is verifying the TLS certificate on HTTPs request.",
			},
			&cli.BoolFlag{
				Destination: &skipDNSCache,
				Value:       skipDNSCache,
				Category:    "Load",
				Name:        "skip-dns-cache",
				Usage: "Skip DNS cache, " +
					"works with \"--url/--hf-*/--ms-*/--ol-*\", " +
					"default is caching the DNS lookup result.",
			},
			&cli.BoolFlag{
				Destination: &skipRangDownloadDetect,
				Value:       skipRangDownloadDetect,
				Category:    "Load",
				Name:        "skip-range-download-detect",
				Aliases: []string{
					"skip-rang-download-detect", // TODO: Fix typo in the next major version
				},
				Usage: "Skip range download detect, " +
					"works with \"--url/--hf-*/--ms-*/--ol-*\", " +
					"default is detecting the range download support.",
			},
			&cli.DurationFlag{
				Destination: &cacheExpiration,
				Value:       cacheExpiration,
				Category:    "Load",
				Name:        "cache-expiration",
				Usage: "Specify the expiration of cache, " +
					"works with \"--url/--hf-*/--ms-*/--ol-*\".",
			},
			&cli.StringFlag{
				Destination: &cachePath,
				Value:       cachePath,
				Category:    "Load",
				Name:        "cache-path",
				Usage: "Cache the read result to the path, " +
					"works with \"--url/--hf-*/--ms-*/--ol-*\".",
			},
			&cli.BoolFlag{
				Destination: &skipCache,
				Value:       skipCache,
				Category:    "Load",
				Name:        "skip-cache",
				Usage: "Skip cache, " +
					"works with \"--url/--hf-*/--ms-*/--ol-*\", " +
					"default is caching the read result.",
			},
			&cli.IntFlag{
				Destination: &parallelSize,
				Value:       parallelSize,
				Category:    "Estimate",
				Name:        "parallel-size",
				Aliases: []string{ // LLaMACpp compatibility
					"parallel",
					"np",
				},
				Usage: "Specify the number of parallel sequences to decode, " +
					"which is used to estimate the usage.",
			},
			&cli.BoolFlag{
				Destination: &flashAttention,
				Value:       flashAttention,
				Category:    "Estimate",
				Name:        "flash-attention",
				Aliases: []string{
					"flash-attn",
					"fa",
					"diffusion-fa", // StableDiffusionCpp compatibility
				},
				Usage: "Specify enabling Flash Attention, " +
					"which is used to estimate the usage. " +
					"Flash Attention can reduce the usage of RAM/VRAM.",
			},
			&cli.BoolFlag{ // LLaMABox compatibility
				Category: "Estimate",
				Name:     "no-flash-attention",
				Aliases: []string{
					"no-flash-attn",
				},
				Usage: "Specify disabling Flash Attention.",
				Action: func(context *cli.Context, b bool) error {
					flashAttention = !b
					return nil
				},
			},
			&cli.UintFlag{
				Destination: &mainGPU,
				Value:       mainGPU,
				Category:    "Estimate",
				Name:        "main-gpu",
				Aliases: []string{ // LLaMACpp compatibility
					"mg",
				},
				Usage: "Specify the GPU to use for the model (with \"--split-mode=none\") " +
					"or for intermediate results and KV (with \"--split-mode=row\"), " +
					"which is used to estimate the usage. " +
					"Since gguf-parser cannot recognize the host GPU devices or RPC servers, " +
					"\"--main-gpu\" only works when \"--tensor-split\" is set.",
			},
			&cli.StringFlag{
				Destination: &rpcServers,
				Value:       rpcServers,
				Category:    "Estimate",
				Name:        "rpc",
				Usage: "Specify the RPC servers, " +
					"which is used to estimate the usage, " +
					"it is a comma-separated list of host:port. " +
					"Woks with \"--tensor-split\".",
			},
			&cli.StringFlag{
				Destination: &tensorSplit,
				Value:       tensorSplit,
				Category:    "Estimate",
				Name:        "tensor-split",
				Aliases: []string{ // LLaMACpp compatibility
					"ts",
				},
				Usage: "Specify the fraction of the model to offload to each device, " +
					"which is used to estimate the usage, " +
					"it is a comma-separated list of integer. " +
					"Since gguf-parser cannot recognize the host GPU devices or RPC servers, " +
					"must explicitly set \"--tensor-split\" to indicate how many devices are used. " +
					"To declare the devices belong to RPC servers, set \"--rpc\" please.",
			},
			&cli.IntFlag{
				Destination: &offloadLayers,
				Value:       offloadLayers,
				Category:    "Estimate",
				Name:        "gpu-layers",
				Aliases: []string{ // LLaMACpp compatibility
					"ngl",
					"n-gpu-layers",
				},
				Usage: "Specify how many layers of the main model to offload, " +
					"which is used to estimate the usage, " +
					"default is full offloaded.",
			},
			&cli.StringSliceFlag{
				Destination: &deviceMetrics,
				Category:    "Estimate",
				Name:        "device-metric",
				Usage: "Specify the device metrics, " +
					"which is used to estimate the throughput, in form of \"FLOPS;Up Bandwidth[;Down Bandwidth]\". " +
					"The FLOPS unit, select from [PFLOPS, TFLOPS, GFLOPS, MFLOPS, KFLOPS]. " +
					"The Up/Down Bandwidth unit, select from [PiBps, TiBps, GiBps, MiBps, KiBps, PBps, TBps, GBps, MBps, KBps, Pbps, Tbps, Gbps, Mbps, Kbps]. " +
					"Up Bandwidth usually indicates the bandwidth to transmit the data to calculate, " +
					"and Down Bandwidth indicates the bandwidth to transmit the calculated result to next layer. " +
					"For example, \"--device-metric 10TFLOPS;400GBps\" means the device has 10 TFLOPS and 400 GBps Up/Down bandwidth, " +
					"\"--device-metric 10TFLOPS;400GBps;5000MBps\" means the device has 5000MBps Down bandwidth. " +
					"If the quantity specified by \"--device-metric\" is less than the number of estimation devices(" +
					"determined by \"--tensor-split\" and \"--rpc\" to infer the device count), " +
					"then replicate the last \"--device-metric\" to meet the required number of evaluation devices.",
			},
			&cli.StringFlag{
				Destination: &platformFootprint,
				Value:       platformFootprint,
				Category:    "Estimate",
				Name:        "platform-footprint",
				Usage: "Specify the platform footprint(RAM,VRAM) of running host in MiB, " +
					"which is used to estimate the NonUMA usage, " +
					"default is \"150,250\". " +
					"Different platform always gets different RAM and VRAM footprints, " +
					"for example, within CUDA, \"cudaMemGetInfo\" or \"cudaSetDevice\" would occupy some RAM and VRAM, " +
					"see https://stackoverflow.com/questions/64854862/free-memory-occupied-by-cudamemgetinfo.",
			},
			&cli.IntFlag{
				Destination: &lmcCtxSize,
				Value:       lmcCtxSize,
				Category:    "Estimate/LLaMACpp",
				Name:        "ctx-size",
				Aliases: []string{ // LLaMACpp compatibility
					"c",
				},
				Usage: "Specify the size of prompt context, " +
					"which is used to estimate the usage, " +
					"default is equal to the model's maximum context size.",
			},
			&cli.BoolFlag{
				Destination: &lmcInMaxCtxSize,
				Value:       lmcInMaxCtxSize,
				Category:    "Estimate/LLaMACpp",
				Name:        "in-max-ctx-size",
				Usage: "Limit the context size to the maximum context size of the model, " +
					"if the context size is larger than the maximum context size.",
			},
			&cli.IntFlag{
				Destination: &lmcLogicalBatchSize,
				Value:       lmcLogicalBatchSize,
				Category:    "Estimate/LLaMACpp",
				Name:        "batch-size",
				Aliases: []string{ // LLaMACpp compatibility
					"b",
				},
				Usage: "Specify the logical batch size, " +
					"which is used to estimate the usage.",
			},
			&cli.IntFlag{
				Destination: &lmcPhysicalBatchSize,
				Value:       lmcPhysicalBatchSize,
				Category:    "Estimate/LLaMACpp",
				Name:        "ubatch-size",
				Aliases: []string{ // LLaMACpp compatibility
					"ub",
				},
				Usage: "Specify the physical maximum batch size, " +
					"which is used to estimate the usage.",
			},
			&cli.StringFlag{
				Destination: &lmcCacheKeyType,
				Value:       lmcCacheKeyType,
				Category:    "Estimate/LLaMACpp",
				Name:        "cache-type-k",
				Aliases: []string{ // LLaMACpp compatibility
					"ctk",
				},
				Usage: "Specify the type of Key cache, " +
					"which is used to estimate the usage, select from [f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1].",
			},
			&cli.StringFlag{
				Destination: &lmcCacheValueType,
				Value:       lmcCacheValueType,
				Category:    "Estimate/LLaMACpp",
				Name:        "cache-type-v",
				Aliases: []string{ // LLaMACpp compatibility
					"ctv",
				},
				Usage: "Specify the type of Value cache, " +
					"which is used to estimate the usage, select from [f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1].",
			},
			&cli.BoolFlag{
				Destination: &lmcNoKVOffload,
				Value:       lmcNoKVOffload,
				Category:    "Estimate/LLaMACpp",
				Name:        "no-kv-offload",
				Aliases: []string{ // LLaMACpp compatibility
					"nkvo",
				},
				Usage: "Specify disabling Key-Value offloading, " +
					"which is used to estimate the usage. " +
					"Disable Key-Value offloading can reduce the usage of VRAM.",
			},
			&cli.StringFlag{
				Destination: &lmcSplitMode,
				Value:       lmcSplitMode,
				Category:    "Estimate/LLaMACpp",
				Name:        "split-mode",
				Aliases: []string{ // LLaMACpp compatibility
					"sm",
				},
				Usage: "Specify how to split the model across multiple devices, " +
					"which is used to estimate the usage, select from [layer, row, none]. " +
					"Since gguf-parser always estimates the usage of VRAM, " +
					"\"none\" is meaningless here, keep for compatibility.",
			},
			&cli.BoolFlag{
				Destination: &lmcNoMMap,
				Value:       lmcNoMMap,
				Category:    "Estimate/LLaMACpp",
				Name:        "no-mmap",
				Usage: "Specify disabling Memory-Mapped using, " +
					"which is used to estimate the usage. " +
					"Memory-Mapped can avoid loading the entire model weights into RAM.",
			},
			&cli.BoolFlag{ // LLaMABox compatibility
				Category: "Estimate/LLaMACpp",
				Name:     "mmap",
				Usage: "Specify enabling Memory-Mapped using, " +
					"which is used to estimate the usage. " +
					"Memory-Mapped can avoid loading the entire model weights into RAM.",
				Action: func(context *cli.Context, b bool) error {
					lmcNoMMap = !b
					return nil
				},
			},
			&cli.UintFlag{ // LLaMABox compatibility
				Destination: &lmcVisualMaxImageSize,
				Value:       lmcVisualMaxImageSize,
				Category:    "Estimate/LLaMACpp",
				Name:        "visual-max-image-size",
				Usage:       "Specify maximum image size when completion with vision model.",
			},
			&cli.IntFlag{
				Destination: &lmcOffloadLayersDraft,
				Value:       lmcOffloadLayersDraft,
				Category:    "Estimate/LLaMACpp",
				Name:        "gpu-layers-draft",
				Aliases: []string{ // LLaMACpp compatibility
					"ngld",
					"n-gpu-layers-draft",
				},
				Usage: "Specify how many layers of the draft model to offload, " +
					"which is used to estimate the usage, " +
					"default is full offloaded.",
			},
			&cli.Uint64Flag{
				Destination: &lmcOffloadLayersStep,
				Value:       lmcOffloadLayersStep,
				Category:    "Estimate/LLaMACpp",
				Name:        "gpu-layers-step",
				Usage: "Specify the step of layers to offload, " +
					"works with \"--gpu-layers\".",
			},
			&cli.UintFlag{
				Destination: &sdcBatchCount,
				Value:       sdcBatchCount,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-batch-count",
				Aliases: []string{
					"batch-count",     // StableDiffusionCpp compatibility
					"image-max-batch", // LLaMABox compatibility
				},
				Usage: "Specify the batch(generation) count of the image.",
			},
			&cli.UintFlag{
				Destination: &sdcHeight,
				Value:       sdcHeight,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-height",
				Aliases: []string{
					"height",           // StableDiffusionCpp compatibility
					"image-max-height", // LLaMABox compatibility
				},
				Usage: "Specify the (maximum) height of the image.",
			},
			&cli.UintFlag{
				Destination: &sdcWidth,
				Value:       sdcWidth,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-width",
				Aliases: []string{
					"width",           // StableDiffusionCpp compatibility
					"image-max-width", // LLaMABox compatibility
				},
				Usage: "Specify the (maximum) width of the image.",
			},
			&cli.BoolFlag{
				Destination: &sdcNoConditionerOffload,
				Value:       sdcNoConditionerOffload,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-no-conditioner-offload",
				Aliases: []string{
					"clip-on-cpu",                         // StableDiffusionCpp compatibility
					"image-no-text-encoder-model-offload", // LLaMABox compatibility
				},
				Usage: "Specify to offload the text encoder model to CPU.",
			},
			&cli.BoolFlag{
				Destination: &sdcNoAutoencoderOffload,
				Value:       sdcNoAutoencoderOffload,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-no-autoencoder-offload",
				Aliases: []string{
					"vae-on-cpu",                 // StableDiffusionCpp compatibility
					"image-no-vae-model-offload", // LLaMABox compatibility
				},
				Usage: "Specify to offload the vae model to CPU.",
			},
			&cli.BoolFlag{
				Destination: &sdcNoControlNetOffload,
				Value:       sdcNoControlNetOffload,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-no-control-net-offload",
				Aliases: []string{
					"control-net-cpu",                    // StableDiffusionCpp compatibility
					"image-no-control-net-model-offload", // LLaMABox compatibility
				},
				Usage: "Specify to offload the control net model to CPU.",
			},
			&cli.BoolFlag{
				Destination: &sdcAutoencoderTiling,
				Value:       sdcAutoencoderTiling,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-autoencoder-tiling",
				Aliases: []string{
					"vae-tiling",       // StableDiffusionCpp compatibility
					"image-vae-tiling", // LLaMABox compatibility
				},
				Usage: "Specify to enable tiling for the vae model.",
			},
			&cli.BoolFlag{
				Destination: &sdcNoAutoencoderTiling,
				Value:       sdcNoAutoencoderTiling,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-no-autoencoder-tiling",
				Aliases: []string{
					"image-no-vae-tiling", // LLaMABox compatibility
				},
				Usage: "Specify to disable tiling for the vae model, it takes precedence over --image-autoencoder-tiling.",
			},
			&cli.BoolFlag{
				Destination: &sdcFreeComputeMemoryImmediately,
				Value:       sdcFreeComputeMemoryImmediately,
				Category:    "Estimate/StableDiffusionCpp",
				Name:        "image-free-compute-memory-immediately", // LLaMABox compatibility
				Usage:       "Specify to free the compute memory immediately after the generation, which burst using VRAM.",
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
				Usage:       "Works with \"--raw\", to save the result to the file",
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
				Usage:       "Works with \"--json\", to output pretty format JSON.",
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
	draftPath            string          // for estimate
	mmprojPath           string          // for estimate
	loraPaths            cli.StringSlice // for estimate
	controlVectorPaths   cli.StringSlice // for estimate
	upscalePath          string          // for estimate
	controlNetPath       string          // for estimate
	url                  string
	draftUrl             string          // for estimate
	mmprojUrl            string          // for estimate
	loraUrls             cli.StringSlice // for estimate
	controlVectorUrls    cli.StringSlice // for estimate
	upscaleUrl           string          // for estimate
	controlNetUrl        string          // for estimate
	token                string
	hfRepo               string
	hfFile               string
	hfDraftRepo          string          // for estimate
	hfDraftFile          string          // for estimate
	hfMMProjFile         string          // for estimate
	hfLoRAFiles          cli.StringSlice // for estimate
	hfControlVectorFiles cli.StringSlice // for estimate
	hfUpscaleRepo        string          // for estimate
	hfUpscaleFile        string          // for estimate
	hfControlNetRepo     string          // for estimate
	hfControlNetFile     string          // for estimate
	hfToken              string
	msRepo               string
	msFile               string
	msDraftRepo          string          // for estimate
	msDraftFile          string          // for estimate
	msMMProjFile         string          // for estimate
	msLoRAFiles          cli.StringSlice // for estimate
	msControlVectorFiles cli.StringSlice // for estimate
	msUpscaleRepo        string          // for estimate
	msUpscaleFile        string          // for estimate
	msControlNetRepo     string          // for estimate
	msControlNetFile     string          // for estimate
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
	parallelSize      = 1
	flashAttention    bool
	mainGPU           uint
	rpcServers        string
	tensorSplit       string
	offloadLayers     = -1
	deviceMetrics     cli.StringSlice
	platformFootprint = "150,250"
	// estimate options for llama.cpp
	lmcCtxSize            = 0
	lmcInMaxCtxSize       bool
	lmcLogicalBatchSize   = 2048
	lmcPhysicalBatchSize  = 512
	lmcCacheKeyType       = "f16"
	lmcCacheValueType     = "f16"
	lmcNoKVOffload        bool
	lmcSplitMode          = "layer"
	lmcNoMMap             bool
	lmcVisualMaxImageSize uint
	lmcOffloadLayersDraft = -1
	lmcOffloadLayersStep  uint64
	// estimate options for stable-diffusion.cpp
	sdcBatchCount                   uint = 1
	sdcHeight                       uint = 1024
	sdcWidth                        uint = 1024
	sdcNoConditionerOffload         bool
	sdcNoAutoencoderOffload         bool
	sdcNoControlNetOffload          bool
	sdcAutoencoderTiling            bool
	sdcNoAutoencoderTiling          bool
	sdcFreeComputeMemoryImmediately bool
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
	if cacheExpiration >= 0 {
		ropts = append(ropts, UseCacheExpiration(cacheExpiration))
	}
	if cachePath != "" {
		ropts = append(ropts, UseCachePath(cachePath))
	}
	if skipCache {
		ropts = append(ropts, SkipCache())
	}

	eopts := []GGUFRunEstimateOption{
		WithLLaMACppCacheValueType(GGMLTypeF16),
		WithLLaMACppCacheKeyType(GGMLTypeF16),
	}
	if parallelSize > 0 {
		eopts = append(eopts, WithParallelSize(int32(parallelSize)))
	}
	if flashAttention {
		eopts = append(eopts, WithFlashAttention())
	}
	if tensorSplit != "" {
		tss := strings.Split(tensorSplit, ",")
		if len(tss) > 128 {
			return errors.New("--tensor-split exceeds the number of devices")
		}
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
	if dmss := deviceMetrics.Value(); len(dmss) > 0 {
		dms := make([]GGUFRunDeviceMetric, len(dmss))
		for i := range dmss {
			ss := strings.Split(dmss[i], ";")
			if len(ss) < 2 {
				return errors.New("--device-metric has invalid format")
			}
			var err error
			dms[i].FLOPS, err = ParseFLOPSScalar(strings.TrimSpace(ss[0]))
			if err != nil {
				return fmt.Errorf("--device-metric has invalid FLOPS: %w", err)
			}
			dms[i].UpBandwidth, err = ParseBytesPerSecondScalar(strings.TrimSpace(ss[1]))
			if err != nil {
				return fmt.Errorf("--device-metric has invalid Up Bandwidth: %w", err)
			}
			if len(ss) > 2 {
				dms[i].DownBandwidth, err = ParseBytesPerSecondScalar(strings.TrimSpace(ss[2]))
				if err != nil {
					return fmt.Errorf("--device-metric has invalid Down Bandwidth: %w", err)
				}
			} else {
				dms[i].DownBandwidth = dms[i].UpBandwidth
			}
		}
		eopts = append(eopts, WithDeviceMetrics(dms))
	}
	if lmcCtxSize > 0 {
		eopts = append(eopts, WithLLaMACppContextSize(int32(lmcCtxSize)))
	}
	if lmcInMaxCtxSize {
		eopts = append(eopts, WithinLLaMACppMaxContextSize())
	}
	if lmcLogicalBatchSize > 0 {
		eopts = append(eopts, WithLLaMACppLogicalBatchSize(int32(max(32, lmcLogicalBatchSize))))
	}
	if lmcPhysicalBatchSize > 0 {
		if lmcPhysicalBatchSize > lmcLogicalBatchSize {
			return errors.New("--ubatch-size must be less than or equal to --batch-size")
		}
		eopts = append(eopts, WithLLaMACppPhysicalBatchSize(int32(lmcPhysicalBatchSize)))
	}
	if lmcCacheKeyType != "" {
		eopts = append(eopts, WithLLaMACppCacheKeyType(toGGMLType(lmcCacheKeyType)))
	}
	if lmcCacheValueType != "" {
		eopts = append(eopts, WithLLaMACppCacheValueType(toGGMLType(lmcCacheValueType)))
	}
	if lmcNoKVOffload {
		eopts = append(eopts, WithoutLLaMACppOffloadKVCache())
	}
	switch lmcSplitMode {
	case "row":
		eopts = append(eopts, WithLLaMACppSplitMode(LLaMACppSplitModeRow))
	case "none":
		eopts = append(eopts, WithLLaMACppSplitMode(LLaMACppSplitModeNone))
	default:
		eopts = append(eopts, WithLLaMACppSplitMode(LLaMACppSplitModeLayer))
	}
	if lmcVisualMaxImageSize > 0 {
		eopts = append(eopts, WithLLaMACppVisualMaxImageSize(uint32(lmcVisualMaxImageSize)))
	}
	if sdcBatchCount > 1 {
		eopts = append(eopts, WithStableDiffusionCppBatchCount(int32(sdcBatchCount)))
	}
	if sdcHeight > 0 {
		eopts = append(eopts, WithStableDiffusionCppHeight(uint32(sdcHeight)))
	}
	if sdcWidth > 0 {
		eopts = append(eopts, WithStableDiffusionCppWidth(uint32(sdcWidth)))
	}
	if sdcNoConditionerOffload {
		eopts = append(eopts, WithoutStableDiffusionCppOffloadConditioner())
	}
	if sdcNoAutoencoderOffload {
		eopts = append(eopts, WithoutStableDiffusionCppOffloadAutoencoder())
	}
	if sdcAutoencoderTiling && !sdcNoAutoencoderTiling {
		eopts = append(eopts, WithStableDiffusionCppAutoencoderTiling())
	}
	if sdcFreeComputeMemoryImmediately {
		eopts = append(eopts, WithStableDiffusionCppFreeComputeMemoryImmediately())
	}
	if offloadLayers >= 0 {
		eopts = append(eopts, WithLLaMACppOffloadLayers(uint64(offloadLayers)), WithStableDiffusionCppOffloadLayers(uint64(offloadLayers)))
	}

	// Parse GGUF file.

	var (
		// Common.
		gf         *GGUFFile
		adapterGfs []*GGUFFile
		// LLaMACpp specific.
		lmcProjectGf *GGUFFile
		lmcDrafterGf *GGUFFile
		// StableDiffusionCpp specific.
		sdcControlNetGf *GGUFFile
		sdcUpscaleGf    *GGUFFile
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
			if err == nil && om != nil && olUsage {
				// Parameters override.
				{
					ps, _ := om.Params(ctx, nil)
					if v, ok := ps["num_ctx"]; ok {
						eopts = append(eopts, WithLLaMACppContextSize(anyx.Number[int32](v)))
					} else if lmcCtxSize <= 0 {
						eopts = append(eopts, WithLLaMACppContextSize(2048))
					}
					if v, ok := ps["use_mmap"]; ok && !anyx.Bool(v) {
						lmcNoMMap = true
					}
					if v, ok := ps["num_gpu"]; ok {
						offloadLayers = anyx.Number[int](v)
					}
				}
				// Multimodal projector overlap.
				{
					mls := om.SearchLayers(regexp.MustCompile(`^application/vnd\.ollama\.image\.projector$`))
					if len(mls) > 0 {
						lmcProjectGf, err = ParseGGUFFileRemote(ctx, mls[len(mls)-1].BlobURL().String(), ropts...)
						if err != nil {
							return fmt.Errorf("failed to parse GGUF file: %w", err)
						}
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
								return fmt.Errorf("failed to parse GGUF file: %w", err)
							}
							adapterGfs = append(adapterGfs, adpgf)
						}
					}
				}
			}
		}
		if err != nil {
			return fmt.Errorf("failed to parse GGUF file: %w", err)
		}

		// Adapter.
		{
			// LoRA.
			for _, loraPath := range loraPaths.Value() {
				adpgf, err := ParseGGUFFile(loraPath, ropts...)
				if err != nil {
					return fmt.Errorf("failed to parse LoRA adapter GGUF file: %w", err)
				}
				adapterGfs = append(adapterGfs, adpgf)
			}
			for _, loraUrl := range loraUrls.Value() {
				adpgf, err := ParseGGUFFileRemote(ctx, loraUrl, ropts...)
				if err != nil {
					return fmt.Errorf("failed to parse LoRA adapter GGUF file: %w", err)
				}
				adapterGfs = append(adapterGfs, adpgf)
			}
			if hfRepo != "" {
				for _, hfLoRAFile := range hfLoRAFiles.Value() {
					adpgf, err := ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfLoRAFile, ropts...)
					if err != nil {
						return fmt.Errorf("failed to parse LoRA adapter GGUF file: %w", err)
					}
					adapterGfs = append(adapterGfs, adpgf)
				}
			}
			if msRepo != "" {
				for _, msLoRAFile := range msLoRAFiles.Value() {
					adpgf, err := ParseGGUFFileFromModelScope(ctx, msRepo, msLoRAFile, ropts...)
					if err != nil {
						return fmt.Errorf("failed to parse LoRA adapter GGUF file: %w", err)
					}
					adapterGfs = append(adapterGfs, adpgf)
				}
			}

			// Control Vector.
			for _, cvPath := range controlVectorPaths.Value() {
				adpgf, err := ParseGGUFFile(cvPath, ropts...)
				if err != nil {
					return fmt.Errorf("failed to parse Control Vector adapter GGUF file: %w", err)
				}
				adapterGfs = append(adapterGfs, adpgf)
			}
			for _, cvUrl := range controlVectorUrls.Value() {
				adpgf, err := ParseGGUFFileRemote(ctx, cvUrl, ropts...)
				if err != nil {
					return fmt.Errorf("failed to parse Control Vector adapter GGUF file: %w", err)
				}
				adapterGfs = append(adapterGfs, adpgf)
			}
			if hfRepo != "" {
				for _, hfCvFile := range hfControlVectorFiles.Value() {
					adpgf, err := ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfCvFile, ropts...)
					if err != nil {
						return fmt.Errorf("failed to parse Control Vector adapter GGUF file: %w", err)
					}
					adapterGfs = append(adapterGfs, adpgf)
				}
			}
			if msRepo != "" {
				for _, msCvFile := range msControlVectorFiles.Value() {
					adpgf, err := ParseGGUFFileFromModelScope(ctx, msRepo, msCvFile, ropts...)
					if err != nil {
						return fmt.Errorf("failed to parse Control Vector adapter GGUF file: %w", err)
					}
					adapterGfs = append(adapterGfs, adpgf)
				}
			}
		}

		// Drafter for LLaMACpp.
		switch {
		case draftPath != "":
			lmcDrafterGf, err = ParseGGUFFile(draftPath, ropts...)
		case draftUrl != "":
			lmcDrafterGf, err = ParseGGUFFileRemote(ctx, draftUrl, ropts...)
		case hfDraftRepo != "" && hfDraftFile != "":
			lmcDrafterGf, err = ParseGGUFFileFromHuggingFace(ctx, hfDraftRepo, hfDraftFile, ropts...)
		case msDraftRepo != "" && msDraftFile != "":
			lmcDrafterGf, err = ParseGGUFFileFromModelScope(ctx, msDraftRepo, msDraftFile, ropts...)
		}
		if err != nil {
			return fmt.Errorf("failed to parse draft GGUF file: %w", err)
		}

		// Projector for LLaMACpp.
		switch {
		case mmprojPath != "":
			lmcProjectGf, err = ParseGGUFFile(mmprojPath, ropts...)
		case mmprojUrl != "":
			lmcProjectGf, err = ParseGGUFFileRemote(ctx, mmprojUrl, ropts...)
		case hfRepo != "" && hfMMProjFile != "":
			lmcProjectGf, err = ParseGGUFFileFromHuggingFace(ctx, hfRepo, hfMMProjFile, ropts...)
		case msRepo != "" && msMMProjFile != "":
			lmcProjectGf, err = ParseGGUFFileFromModelScope(ctx, msRepo, msMMProjFile, ropts...)
		}
		if err != nil {
			return fmt.Errorf("failed to parse multimodal projector GGUF file: %w", err)
		}

		// ControlNet for StableDiffusionCpp.
		switch {
		case controlNetPath != "":
			sdcControlNetGf, err = ParseGGUFFile(controlNetPath, ropts...)
		case controlNetUrl != "":
			sdcControlNetGf, err = ParseGGUFFileRemote(ctx, controlNetUrl, ropts...)
		case hfControlNetRepo != "" && hfControlNetFile != "":
			sdcControlNetGf, err = ParseGGUFFileFromHuggingFace(ctx, hfControlNetRepo, hfControlNetFile, ropts...)
		case msControlNetRepo != "" && msControlNetFile != "":
			sdcControlNetGf, err = ParseGGUFFileFromModelScope(ctx, msControlNetRepo, msControlNetFile, ropts...)
		}
		if err != nil {
			return fmt.Errorf("failed to parse control net GGUF file: %w", err)
		}

		// Upscaler for StableDiffusionCpp.
		switch {
		case upscalePath != "":
			sdcUpscaleGf, err = ParseGGUFFile(upscalePath, ropts...)
		case upscaleUrl != "":
			sdcUpscaleGf, err = ParseGGUFFileRemote(ctx, upscaleUrl, ropts...)
		case hfUpscaleRepo != "" && hfUpscaleFile != "":
			sdcUpscaleGf, err = ParseGGUFFileFromHuggingFace(ctx, hfUpscaleRepo, hfUpscaleFile, ropts...)
		case msUpscaleRepo != "" && msUpscaleFile != "":
			sdcUpscaleGf, err = ParseGGUFFileFromModelScope(ctx, msUpscaleRepo, msUpscaleFile, ropts...)
		}
		if err != nil {
			return fmt.Errorf("failed to parse upscaler GGUF file: %w", err)
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
		m   = gf.Metadata()
		a   = gf.Architecture()
		t   = gf.Tokenizer()
		lme LLaMACppRunEstimate
		sde StableDiffusionCppRunEstimate
	)

	skipTokenizer = skipTokenizer || t.Model == ""
	skipEstimate = skipEstimate || m.Type != "model"

	if !skipEstimate && m.Architecture != "diffusion" {
		if lmcDrafterGf != nil {
			dlmceopts := eopts[:len(eopts):len(eopts)]
			if lmcOffloadLayersDraft >= 0 {
				dlmceopts = append(dlmceopts, WithLLaMACppOffloadLayers(uint64(lmcOffloadLayersDraft)))
			}
			dlmceopts = append(dlmceopts, WithLLaMACppCacheKeyType(GGMLTypeF16), WithLLaMACppCacheValueType(GGMLTypeF16))
			de := lmcDrafterGf.EstimateLLaMACppRun(dlmceopts...)
			eopts = append(eopts, WithLLaMACppDrafter(&de))
		}

		if lmcProjectGf != nil {
			plmceopts := eopts[:len(eopts):len(eopts)]
			me := lmcProjectGf.EstimateLLaMACppRun(plmceopts...)
			eopts = append(eopts, WithLLaMACppProjector(&me))
		}

		if len(adapterGfs) > 0 {
			adps := make([]LLaMACppRunEstimate, len(adapterGfs))
			almceopts := eopts[:len(eopts):len(eopts)]
			for i, adpgf := range adapterGfs {
				ae := adpgf.EstimateLLaMACppRun(almceopts...)
				adps[i] = ae
			}
			eopts = append(eopts, WithLLaMACppAdapters(adps))
		}

		lme = gf.EstimateLLaMACppRun(eopts...)
	}

	if !skipEstimate && m.Architecture == "diffusion" {
		if sdcUpscaleGf != nil {
			sdceopts := eopts[:len(eopts):len(eopts)]
			ue := sdcUpscaleGf.EstimateStableDiffusionCppRun(sdceopts...)
			eopts = append(eopts, WithStableDiffusionCppUpscaler(&ue))
		}

		if sdcControlNetGf != nil {
			sdceopts := eopts[:len(eopts):len(eopts)]
			if sdcNoControlNetOffload {
				sdceopts = append(sdceopts, WithStableDiffusionCppOffloadLayers(0))
			}
			ce := sdcControlNetGf.EstimateStableDiffusionCppRun(sdceopts...)
			eopts = append(eopts, WithStableDiffusionCppControlNet(&ce))
		}

		sde = gf.EstimateStableDiffusionCppRun(eopts...)
	}

	// Then, output as JSON or table.

	var (
		mmap                      = !lmcNoMMap
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

		if !skipTokenizer {
			o["tokenizer"] = t
		}

		if !skipEstimate && m.Architecture != "diffusion" {
			lmes := lme.Summarize(mmap, platformRAM, platformVRAM)
			switch {
			case lmcOffloadLayersStep > lme.OffloadLayers:
				lmcOffloadLayersStep = lme.OffloadLayers
			case lmcOffloadLayersStep <= 0:
				lmcOffloadLayersStep = lme.OffloadLayers
			}
			if lmcOffloadLayersStep < lme.OffloadLayers {
				cnt := lme.OffloadLayers/lmcOffloadLayersStep + 1
				if lme.OffloadLayers%lmcOffloadLayersStep != 0 || lme.FullOffloaded {
					cnt++
				}
				esis := make([]LLaMACppRunEstimateSummaryItem, cnt)
				var wg sync.WaitGroup
				for i := 0; i < cap(esis); i++ {
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						lmeopts := eopts[:len(eopts):len(eopts)]
						lmeopts = append(lmeopts, WithLLaMACppOffloadLayers(uint64(i)*lmcOffloadLayersStep))
						esis[i] = gf.EstimateLLaMACppRun(lmeopts...).SummarizeItem(mmap, platformRAM, platformVRAM)
					}(i)
				}
				wg.Wait()
				esis[cap(esis)-1] = lmes.Items[0]
				lmes.Items = esis
			}
			o["estimate"] = lmes
		}

		if !skipEstimate && m.Architecture == "diffusion" {
			sdes := sde.Summarize(mmap, platformRAM, platformVRAM)
			o["estimate"] = sdes
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

	GGUFBytesScalarStringInMiBytes = inMib

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
			if a.Architecture == "diffusion" {
				hd = []any{
					"Diffusion Arch",
					"Conditioners",
					"Autoencoder",
				}
				bd = []any{
					a.DiffusionArchitecture,
					sprintf(tenary(a.DiffusionHasConditioners(), a.DiffusionConditioners, "N/A")),
					sprintf(tenary(a.DiffusionHasAutoencoder(), a.DiffusionAutoencoder, "N/A")),
				}
			} else {
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
		}
		tprint(
			"ARCHITECTURE",
			[][]any{hd},
			[][]any{bd})
	}

	if !skipTokenizer {
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

	if !skipEstimate && m.Architecture != "diffusion" {
		hds := make([][]any, 2)
		lmes := lme.Summarize(mmap, platformRAM, platformVRAM)
		if !inShort {
			hds[0] = []any{
				"Arch",
				"Context Size",
				"Batch Size (L / P)",
				"Flash Attention",
				"MMap Load",
				"Embedding Only",
				"Reranking",
				"Distributable",
				"Offload Layers",
				"Full Offloaded",
			}
			hds[1] = []any{
				"Arch",
				"Context Size",
				"Batch Size (L / P)",
				"Flash Attention",
				"MMap Load",
				"Embedding Only",
				"Reranking",
				"Distributable",
				"Offload Layers",
				"Full Offloaded",
			}
		}
		if lmes.Items[0].MaximumTokensPerSecond != nil {
			hds[0] = append(hds[0], "Max TPS")
			hds[1] = append(hds[1], "Max TPS")
		}
		hds[0] = append(hds[0], "RAM", "RAM", "RAM")
		hds[1] = append(hds[1], "Layers (I + T + O)", "UMA", "NonUMA")
		for _, v := range lmes.Items[0].VRAMs {
			var hd string
			if v.Remote {
				hd = fmt.Sprintf("RPC %d (V)RAM", v.Position)
			} else {
				hd = fmt.Sprintf("VRAM %d", v.Position)
			}
			hds[0] = append(hds[0], hd, hd, hd)
			hds[1] = append(hds[1], "Layers (T + O)", "UMA", "NonUMA")
		}

		switch {
		case lmcOffloadLayersStep > lme.OffloadLayers:
			lmcOffloadLayersStep = lme.OffloadLayers
		case lmcOffloadLayersStep <= 0:
			lmcOffloadLayersStep = lme.OffloadLayers
		}
		if lmcOffloadLayersStep < lme.OffloadLayers {
			cnt := lme.OffloadLayers/lmcOffloadLayersStep + 1
			if lme.OffloadLayers%lmcOffloadLayersStep != 0 || lme.FullOffloaded {
				cnt++
			}
			esis := make([]LLaMACppRunEstimateSummaryItem, cnt)
			var wg sync.WaitGroup
			for i := 0; i < cap(esis); i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					lmeopts := eopts[:len(eopts):len(eopts)]
					lmeopts = append(lmeopts, WithLLaMACppOffloadLayers(uint64(i)*lmcOffloadLayersStep))
					esis[i] = gf.EstimateLLaMACppRun(lmeopts...).SummarizeItem(mmap, platformRAM, platformVRAM)
				}(i)
			}
			wg.Wait()
			esis[cap(esis)-1] = lmes.Items[0]
			lmes.Items = esis
		}

		bds := make([][]any, len(lmes.Items))
		for i := range lmes.Items {
			if !inShort {
				bds[i] = []any{
					sprintf(lmes.Architecture),
					sprintf(lmes.ContextSize),
					sprintf("%d / %d", lmes.LogicalBatchSize, lmes.PhysicalBatchSize),
					sprintf(tenary(flashAttention, tenary(lmes.FlashAttention, "Enabled", "Unsupported"), "Disabled")),
					sprintf(tenary(mmap, tenary(!lmes.NoMMap, "Enabled", "Unsupported"), "Disabled")),
					sprintf(tenary(lmes.EmbeddingOnly, "Yes", "No")),
					sprintf(tenary(lmes.Reranking, "Supported", "Unsupported")),
					sprintf(tenary(lmes.Distributable, "Supported", "Unsupported")),
					sprintf(tenary(lmes.Items[i].FullOffloaded, sprintf("%d (%d + 1)",
						lmes.Items[i].OffloadLayers, lmes.Items[i].OffloadLayers-1), lmes.Items[i].OffloadLayers)),
					sprintf(tenary(lmes.Items[i].FullOffloaded, "Yes", "No")),
				}
			}
			if lmes.Items[i].MaximumTokensPerSecond != nil {
				bds[i] = append(bds[i],
					sprintf(*lmes.Items[i].MaximumTokensPerSecond))
			}
			bds[i] = append(bds[i],
				sprintf("1 + %d + %d", lmes.Items[i].RAM.HandleLayers, tenary(lmes.Items[i].RAM.HandleOutputLayer, 1, 0)),
				sprintf(lmes.Items[i].RAM.UMA),
				sprintf(lmes.Items[i].RAM.NonUMA))
			for _, v := range lmes.Items[i].VRAMs {
				bds[i] = append(bds[i],
					sprintf("%d + %d", v.HandleLayers, tenary(v.HandleOutputLayer, 1, 0)),
					sprintf(v.UMA),
					sprintf(v.NonUMA))
			}
		}

		tprint(
			"ESTIMATE",
			hds,
			bds)
	}

	if !skipEstimate && m.Architecture == "diffusion" {
		hds := make([][]any, 2)
		sdes := sde.Summarize(mmap, platformRAM, platformVRAM)
		if !inShort {
			hds[0] = []any{
				"Arch",
				"Flash Attention",
				"MMap Load",
				"Distributable",
				"Full Offloaded",
			}
			hds[1] = []any{
				"Arch",
				"Flash Attention",
				"MMap Load",
				"Distributable",
				"Full Offloaded",
			}
		}
		hds[0] = append(hds[0], "RAM", "RAM")
		hds[1] = append(hds[1], "UMA", "NonUMA")
		for _, v := range sdes.Items[0].VRAMs {
			var hd string
			if v.Remote {
				hd = fmt.Sprintf("RPC %d (V)RAM", v.Position)
			} else {
				hd = fmt.Sprintf("VRAM %d", v.Position)
			}
			hds[0] = append(hds[0], hd, hd)
			hds[1] = append(hds[1], "UMA", "NonUMA")
		}

		bds := make([][]any, len(sdes.Items))
		for i := range sdes.Items {
			if !inShort {
				bds[i] = []any{
					sprintf(sdes.Architecture),
					sprintf(tenary(flashAttention, tenary(sdes.FlashAttention, "Enabled", "Unsupported"), "Disabled")),
					sprintf(tenary(mmap, tenary(!sdes.NoMMap, "Enabled", "Unsupported"), "Disabled")),
					sprintf(tenary(sdes.Distributable, "Supported", "Unsupported")),
					sprintf(tenary(sdes.Items[i].FullOffloaded, "Yes", "No")),
				}
			}
			bds[i] = append(bds[i],
				sprintf(sdes.Items[i].RAM.UMA),
				sprintf(sdes.Items[i].RAM.NonUMA))
			for _, v := range sdes.Items[i].VRAMs {
				bds[i] = append(bds[i],
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
			if len(headers) > 1 && (strings.HasPrefix(headers[1][i].(string), "Layers") || headers[1][i] == "UMA" || headers[1][i] == "NonUMA") {
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
