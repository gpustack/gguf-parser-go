# GGUF Parser

Review/Check/Estimate [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) file.

## Usage

```shell
$ gguf-parser --help
Usage of gguf-parser ...:
  -ctx-size int
        Specify the size of prompt context, which is used to estimate the usage, default is equal to the model's maximum context size. (default -1)
  -debug
        Enable debugging, verbosity.
  -file string
        [DEPRECATED, use --hf-file instead] Model file below the --repo, e.g. Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.
  -flash-attention
        Specify enabling Flash Attention, which is used to estimate the usage. Flash Attention can reduce the usage of RAM/VRAM.
  -gpu-layers int
        Specify how many layers to offload, which is used to estimate the usage, default is full offloaded. (default -1)
  -gpu-layers-step uint
        Specify the step of layers to offload, works with --gpu-layers.
  -hf-file string
        Model file below the --hf-repo, e.g. Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.
  -hf-repo string
        Repository of HuggingFace which the GGUF file store, e.g. NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF, works with --hf-file.
  -in-max-ctx-size
        Limit the context size to the maximum context size of the model, if the context size is larger than the maximum context size.
  -in-mib
        Display the estimated result in table with MiB.
  -json
        Output as JSON.
  -json-pretty
        Output as pretty JSON. (default true)
  -kv-type string
        Specify the type of Key-Value cache, which is used to estimate the usage, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1], default is f16. Use quantization type means enabling --flash-attention as well. (default "f16")
  -ms-file string
        Model file below the --ms-repo, e.g. qwen1.5-0.5b-chat.gguf.
  -ms-repo string
        Repository of ModelScope which the GGUF file store, e.g. qwen/Qwen1.5-0.5B-Chat-GGUF, works with --ms-file.
  -no-kv-offload
        Specify disabling Key-Value offloading, which is used to estimate the usage. Key-Value offloading can reduce the usage of VRAM.
  -no-mmap
        Specify disabling Memory-Mapped using, which is used to estimate the usage. Memory-Mapped can avoid loading the entire model weights into RAM.
  -offload-layers int
        [DEPRECATED, use --gpu-layers instead] Specify how many layers to offload, which is used to estimate the usage, default is full offloaded. (default -1)
  -offload-layers-step uint
        [DEPRECATED, use --gpu-layers-step instead] Specify the step of layers to offload, works with --offload-layers.
  -ol-crawl
        Crawl the Ollama model instead of blobs fetching, works with --ol-model, which will be more efficient and faster, but lossy.
  -ol-model string
        Model name of Ollama, e.g. gemma2.
  -ol-usage
        Specify respecting the extending layers introduced by Ollama, works with --ol-model, which affects the usage estimation.
  -parallel-size int
        Specify the number of parallel sequences to decode, which is used to estimate the usage, default is 1. (default 1)
  -path string
        Path where the GGUF file to load, e.g. ~/.cache/lm-studio/models/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.
  -platform-footprint cudaMemGetInfo
        Specify the platform footprint(RAM,VRAM) in MiB, which is used to estimate the NonUMA usage, default is 150,250. Different platform always gets different RAM and VRAM footprints, for example, within CUDA, cudaMemGetInfo would occupy some RAM and VRAM, see https://stackoverflow.com/questions/64854862/free-memory-occupied-by-cudamemgetinfo. (default "150,250")
  -raw
        Output the file only, skip anything.
  -repo string
        [DEPRECATED, use --hf-repo instead] Repository of HuggingFace which the GGUF file store, e.g. NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF, works with --file.
  -skip-architecture
        Skip to display architecture metadata.
  -skip-cache
        Skip cache, works with --url/--hf-*/--ms-*/--ol-*, default is caching the read result.
  -skip-dns-cache
        Skip DNS cache, works with --url/--hf-*/--ms-*/--ol-*, default is caching the DNS lookup result.
  -skip-estimate
        Skip to estimate.
  -skip-model
        Skip to display model metadata.
  -skip-proxy
        Skip proxy settings, works with --url/--hf-*/--ms-*/--ol-*, default is respecting the environment variables HTTP_PROXY/HTTPS_PROXY/NO_PROXY.
  -skip-rang-download-detect
        Skip range download detect, works with --url/--hf-*/--ms-*/--ol-*, default is detecting the range download support.
  -skip-tls-verify
        Skip TLS verification, works with --url/--hf-*/--ms-*/--ol-*, default is verifying the TLS certificate on HTTPs request.
  -skip-tokenizer
        Skip to display tokenizer metadata
  -ubatch-size int
        Specify the physical maximum batch size, which is used to estimate the usage, default is 512. (default 512)
  -url string
        Url where the GGUF file to load, e.g. https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf. Note that gguf-parser does not need to download the entire GGUF file.
  -version
        Show gguf-parser version.

```

### Parse

#### Parse local GGUF file

```shell
$ gguf-parser --path="~/.cache/lm-studio/models/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF/Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf"
+--------------+-------+-------+----------------+---------------+----------+------------+----------+
|      \       | Name  | Arch  |  Quantization  | Little Endian |   Size   | Parameters |   BPW    |
+--------------+-------+-------+----------------+---------------+----------+------------+----------+
|    MODEL     | jeffq | llama | IQ3_XXS/Q5_K_M |     true      | 4.78 GiB |   7.24 B   | 5.67 bpw |
+--------------+-------+-------+----------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      32768      |     4096      |       4       |         32         |   32   |      14336       |     0      |     32032      |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama | 450.50 KiB  |   32032    |       N/A        |     1     |   32000   |      N/A      |       N/A       |      N/A      |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+--------------+-------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |        UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |      false      |     true     |  33 (32 + 1)   |      Yes       | 88.39 MiB + 8.59 GiB = 8.68 GiB | 238.39 MiB |  11.06 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+

```

#### Parse remote GGUF file

```shell
$ gguf-parser --url="https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/resolve/main/Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf"
+--------------+----------+-------+--------------+---------------+--------+------------+----------+
|      \       |   Name   | Arch  | Quantization | Little Endian |  Size  | Parameters |   BPW    |
+--------------+----------+-------+--------------+---------------+--------+------------+----------+
|    MODEL     | emozilla | llama | Q4_K/Q3_K_M  |     true      | 21 GiB |  46.70 B   | 3.86 bpw |
+--------------+----------+-------+--------------+---------------+--------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      32768      |     4096      |       4       |         32         |   32   |      14336       |     8      |     32002      |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama | 449.91 KiB  |   32002    |       N/A        |     1     |   32000   |       0       |       N/A       |       2       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |      false      |    false     |  33 (32 + 1)   |      Yes       | 245.24 MiB + 24.84 GiB = 25.08 GiB | 395.24 MiB |  27.31 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Parse HuggingFace GGUF file

```shell
$ gguf-parser --hf-repo="openbmb/MiniCPM-Llama3-V-2_5-gguf" --hf-file="ggml-model-Q5_K_M.gguf" 
+--------------+-------+-------+----------------+---------------+----------+------------+----------+
|      \       | Name  | Arch  |  Quantization  | Little Endian |   Size   | Parameters |   BPW    |
+--------------+-------+-------+----------------+---------------+----------+------------+----------+
|    MODEL     | model | llama | IQ3_XXS/Q5_K_M |     true      | 5.33 GiB |   8.03 B   | 5.70 bpw |
+--------------+-------+-------+----------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      8192       |     4096      |       4       |         32         |   32   |      14336       |     0      |     128256     |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | gpt2  |    2 MiB    |   128256   |       N/A        |  128000   |  128001   |    128002     |       N/A       |       0       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+--------------+-------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |        UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+
|   ESTIMATE   | llama |     8192     |      false      |     true     |  33 (32 + 1)   |      Yes       | 84.61 MiB + 5.59 GiB = 5.68 GiB | 234.61 MiB |  6.49 GiB   |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+

```

#### Parse ModelScope GGUF file

```shell
$ gguf-parser --ms-repo="shaowenchen/chinese-alpaca-2-13b-16k-gguf" --ms-file="chinese-alpaca-2-13b-16k.Q5_K.gguf"
+--------------+------+-------+----------------+---------------+----------+------------+----------+
|      \       | Name | Arch  |  Quantization  | Little Endian |   Size   | Parameters |   BPW    |
+--------------+------+-------+----------------+---------------+----------+------------+----------+
|    MODEL     |  ..  | llama | IQ3_XXS/Q5_K_M |     true      | 8.76 GiB |  13.25 B   | 5.68 bpw |
+--------------+------+-------+----------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      16384      |     5120      |       1       |        N/A         |   40   |      13824       |     0      |     55296      |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama | 769.83 KiB  |   55296    |       N/A        |     1     |     2     |      N/A      |       N/A       |      N/A      |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+--------------+-------+--------------+-----------------+--------------+----------------+----------------+-----------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |         UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+-----------------------------------+------------+-------------+
|   ESTIMATE   | llama |    16384     |      false      |     true     |  41 (40 + 1)   |      Yes       | 61.18 MiB + 20.87 GiB = 20.92 GiB | 211.18 MiB |  22.74 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+-----------------------------------+------------+-------------+

```

#### Parse Ollama model

```shell
$ gguf-parser --ol-model="gemma2"
+--------------+--------+--------+--------------+---------------+----------+------------+----------+
|      \       |  Name  |  Arch  | Quantization | Little Endian |   Size   | Parameters |   BPW    |
+--------------+--------+--------+--------------+---------------+----------+------------+----------+
|    MODEL     | gemma2 | gemma2 |     Q4_0     |     true      | 5.06 GiB |   9.24 B   | 4.71 bpw |
+--------------+--------+--------+--------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      8192       |     3584      |       2       |         16         |   42   |      14336       |     0      |     256000     |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama |  3.80 MiB   |   256000   |       N/A        |     2     |     1     |       3       |       N/A       |       0       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+--------------+--------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+
|      \       |  Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |        UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+--------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+
|   ESTIMATE   | gemma2 |     8192     |      false      |     true     |  43 (42 + 1)   |      Yes       | 65.97 MiB + 6.99 GiB = 7.05 GiB | 215.97 MiB |  8.43 GiB   |
+--------------+--------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+

$ gguf-parser --ol-model="gemma2" --ol-crawl
+--------------+--------+--------+--------------+---------------+----------+------------+----------+
|      \       |  Name  |  Arch  | Quantization | Little Endian |   Size   | Parameters |   BPW    |
+--------------+--------+--------+--------------+---------------+----------+------------+----------+
|    MODEL     | gemma2 | gemma2 |     Q4_0     |     true      | 5.06 GiB |   9.24 B   | 4.71 bpw |
+--------------+--------+--------+--------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      8192       |     3584      |       2       |         16         |   42   |      14336       |     0      |     256000     |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama |     N/A     |   256000   |       N/A        |     2     |     1     |       3       |       N/A       |       0       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+--------------+--------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+
|      \       |  Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |        UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+--------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+
|   ESTIMATE   | gemma2 |     8192     |      false      |     true     |  43 (42 + 1)   |      Yes       | 65.99 MiB + 6.99 GiB = 7.05 GiB | 215.99 MiB |  8.43 GiB   |
+--------------+--------+--------------+-----------------+--------------+----------------+----------------+---------------------------------+------------+-------------+

```

#### Parse Clip model

```shell
$ gguf-parser --hf-repo="xtuner/llava-llama-3-8b-v1_1-gguf" --hf-file="llava-llama-3-8b-v1_1-mmproj-f16.gguf"
+--------------+-----------------------------------+------+--------------+---------------+------------+------------+-----------+
|      \       |               Name                | Arch | Quantization | Little Endian |    Size    | Parameters |    BPW    |
+--------------+-----------------------------------+------+--------------+---------------+------------+------------+-----------+
|    MODEL     | openai/clip-vit-large-patch14-336 | clip |     F16      |     true      | 595.49 MiB |  311.89 M  | 16.02 bpw |
+--------------+-----------------------------------+------+--------------+---------------+------------+------------+-----------+

+--------------+---------------+--------+------------------+---------+-----------------+
|      \       | Embedding Len | Layers | Feed Forward Len | Encoder | LLaVA Projector |
+--------------+---------------+--------+------------------+---------+-----------------+
| ARCHITECTURE |     1024      |   23   |       4096       | Vision  |       mlp       |
+--------------+---------------+--------+------------------+---------+-----------------+

+--------------+------+----------------+----------------+------------+
|      \       | Arch | Offload Layers | Full Offloaded |   (V)RAM   |
+--------------+------+----------------+----------------+------------+
|   ESTIMATE   | clip |       24       |      Yes       | 595.49 MiB |
+--------------+------+----------------+----------------+------------+

```

### Estimate

#### Estimate with full layers offload (default)

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |      false      |    false     |  33 (32 + 1)   |      Yes       | 245.24 MiB + 24.84 GiB = 25.08 GiB | 395.24 MiB |  27.31 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Estimate with zero layers offload

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --gpu-layers=0
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+-----------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |      UMA (RAM + VRAM)       | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+-----------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |      false      |    false     |       0        |       No       | 25.09 GiB + 0 B = 25.09 GiB | 25.24 GiB  |  2.46 GiB   |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+-----------------------------+------------+-------------+

```

#### Estimate with specific layers offload

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --gpu-layers=10
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |     4096     |      false      |    false     |  33 (32 + 1)   |      Yes       | 189.24 MiB + 21.34 GiB = 21.53 GiB | 339.24 MiB |  21.89 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Estimate with specific context size

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --ctx-size=4096
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |     4096     |      false      |    false     |  33 (32 + 1)   |      Yes       | 189.24 MiB + 21.34 GiB = 21.53 GiB | 339.24 MiB |  21.89 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Estimate with Flash Attention

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --flash-attention
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |      true       |    false     |  33 (32 + 1)   |      Yes       | 245.24 MiB + 24.84 GiB = 25.08 GiB | 395.24 MiB |  25.33 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Estimate with No MMap

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --offload-layers=10 --no-mmap
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+----------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |         UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+----------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |      false      |    false     |       10       |       No       | 17.36 GiB + 7.73 GiB = 25.09 GiB | 17.51 GiB  |  10.19 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+----------------------------------+------------+-------------+

```

#### Estimate step-by-step offload layers

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --gpu-layers-step=5
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Flash Attention | MMap Support | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |      false      |    false     |       0        |       No       |    25.09 GiB + 0 B = 25.09 GiB     | 25.24 GiB  |  2.46 GiB   |
+              +       +              +                 +              +----------------+                +------------------------------------+------------+-------------+
|              |       |              |                 |              |       5        |                |  21.23 GiB + 3.86 GiB = 25.09 GiB  | 21.37 GiB  |  6.33 GiB   |
+              +       +              +                 +              +----------------+                +------------------------------------+------------+-------------+
|              |       |              |                 |              |       10       |                |  17.36 GiB + 7.73 GiB = 25.09 GiB  | 17.51 GiB  |  10.19 GiB  |
+              +       +              +                 +              +----------------+                +------------------------------------+------------+-------------+
|              |       |              |                 |              |       15       |                | 13.50 GiB + 11.59 GiB = 25.09 GiB  | 13.64 GiB  |  14.06 GiB  |
+              +       +              +                 +              +----------------+                +------------------------------------+------------+-------------+
|              |       |              |                 |              |       20       |                |  9.63 GiB + 15.46 GiB = 25.09 GiB  |  9.78 GiB  |  17.92 GiB  |
+              +       +              +                 +              +----------------+                +------------------------------------+------------+-------------+
|              |       |              |                 |              |       25       |                |  5.77 GiB + 19.32 GiB = 25.09 GiB  |  5.91 GiB  |  21.79 GiB  |
+              +       +              +                 +              +----------------+                +------------------------------------+------------+-------------+
|              |       |              |                 |              |       30       |                |  1.90 GiB + 23.19 GiB = 25.09 GiB  |  2.05 GiB  |  25.65 GiB  |
+              +       +              +                 +              +----------------+----------------+------------------------------------+------------+-------------+
|              |       |              |                 |              |  33 (32 + 1)   |      Yes       | 245.24 MiB + 24.84 GiB = 25.08 GiB | 395.24 MiB |  27.31 GiB  |
+--------------+-------+--------------+-----------------+--------------+----------------+----------------+------------------------------------+------------+-------------+

```

## License

MIT
