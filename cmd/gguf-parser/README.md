# GGUF Parser

Review/Check/Estimate [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) file.

## Usage

```shell
$ gguf-parser --help
NAME:
   gguf-parser - Review/Check/Estimate the GGUF file.

USAGE:
   gguf-parser [GLOBAL OPTIONS]

GLOBAL OPTIONS:
   --debug        Enable debugging, verbosity. (default: false)
   --help, -h     Print the usage.
   --version, -v  Print the version.

   Estimate

   --batch-size value, -b value                         Specify the logical batch size, which is used to estimate the usage. (default: 2048)
   --cache-type-k value, --ctk value                    Specify the type of Key cache, which is used to estimate the usage, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1]. (default: "f16")
   --cache-type-v value, --ctv value                    Specify the type of Value cache, which is used to estimate the usage, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1]. (default: "f16")
   --ctx-size value, -c value                           Specify the size of prompt context, which is used to estimate the usage, default is equal to the model's maximum context size. (default: -1)
   --flash-attention, --flash-attn, --fa                Specify enabling Flash Attention, which is used to estimate the usage. Flash Attention can reduce the usage of RAM/VRAM. (default: false)
   --gpu-layers value, --ngl value                      Specify how many layers of the main model to offload, which is used to estimate the usage, default is full offloaded. (default: -1)
   --gpu-layers-draft value, --ngld value               Specify how many layers of the draft model to offload, which is used to estimate the usage, default is full offloaded. (default: -1)
   --gpu-layers-step value                              Specify the step of layers to offload, works with --gpu-layers. (default: 0)
   --in-max-ctx-size                                    Limit the context size to the maximum context size of the model, if the context size is larger than the maximum context size. (default: false)
   --kv-type value                                      Specify the type of Key-Value cache, which is used to estimate the usage, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1]. (Deprecated, use --cache-type-k/--cache-type-v instead, will be removed after v0.7.0) (default: "f16")
   --no-kv-offload, --nkvo                              Specify disabling Key-Value offloading, which is used to estimate the usage. Disable Key-Value offloading can reduce the usage of VRAM. (default: false)
   --no-mmap                                            Specify disabling Memory-Mapped using, which is used to estimate the usage. Memory-Mapped can avoid loading the entire model weights into RAM. (default: false)
   --parallel-size value, --parallel value, --np value  Specify the number of parallel sequences to decode, which is used to estimate the usage. (default: 1)
   --platform-footprint value                           Specify the platform footprint(RAM,VRAM) in MiB, which is used to estimate the NonUMA usage, default is 150,250. Different platform always gets different RAM and VRAM footprints, for example, within CUDA, 'cudaMemGetInfo' would occupy some RAM and VRAM, see https://stackoverflow.com/questions/64854862/free-memory-occupied-by-cudamemgetinfo. (default: "150,250")
   --ubatch-size value, --ub value                      Specify the physical maximum batch size, which is used to estimate the usage. (default: 512)

   Load

   --skip-cache                 Skip cache, works with --url/--hf-*/--ms-*/--ol-*, default is caching the read result. (default: false)
   --skip-dns-cache             Skip DNS cache, works with --url/--hf-*/--ms-*/--ol-*, default is caching the DNS lookup result. (default: false)
   --skip-proxy                 Skip proxy settings, works with --url/--hf-*/--ms-*/--ol-*, default is respecting the environment variables HTTP_PROXY/HTTPS_PROXY/NO_PROXY. (default: false)
   --skip-rang-download-detect  Skip range download detect, works with --url/--hf-*/--ms-*/--ol-*, default is detecting the range download support. (default: false)
   --skip-tls-verify            Skip TLS verification, works with --url/--hf-*/--ms-*/--ol-*, default is verifying the TLS certificate on HTTPs request. (default: false)

   Model/Local

   --draft-path value, --model-draft value, --md value  Path where the GGUF file to load for the draft model, optional, e.g. ~/.cache/lm-studio/models/QuantFactory/Qwen2-1.5B-Instruct-GGUF/Qwen2-1.5B-Instruct.Q5_K_M.gguf
   --mmproj-path value, --mmproj value                  Path where the GGUF file to load for the multimodal projector, optional.
   --path value, --model value, -m value                Path where the GGUF file to load for the main model, e.g. ~/.cache/lm-studio/models/QuantFactory/Qwen2-7B-Instruct-GGUF/Qwen2-7B-Instruct.Q5_K_M.gguf.

   Model/Remote

   --draft-url value                           Url where the GGUF file to load for the draft model, optional, e.g. https://huggingface.co/QuantFactory/Qwen2-1.5B-Instruct-GGUF/resolve/main/Qwen2-1.5B-Instruct.Q5_K_M.gguf. Note that gguf-parser does not need to download the entire GGUF file.
   --mmproj-url value                          Url where the GGUF file to load for the multimodal projector, optional.
   --token value                               Bearer auth token to load GGUF file, optional, works with --url/--draft-url.
   --url value, --model-url value, --mu value  Url where the GGUF file to load for the main model, e.g. https://huggingface.co/QuantFactory/Qwen2-7B-Instruct-GGUF/resolve/main/Qwen2-7B-Instruct.Q5_K_M.gguf. Note that gguf-parser does not need to download the entire GGUF file.

   Model/Remote/HuggingFace

   --hf-draft-file value          Model file below the --hf-draft-repo, optional, e.g. Qwen2-1.5B-Instruct.Q5_K_M.gguf.
   --hf-draft-repo value          Repository of HuggingFace which the GGUF file store for the draft model, optional, e.g. QuantFactory/Qwen2-1.5B-Instruct-GGUF, works with --hf-draft-file.
   --hf-file value, --hff value   Model file below the --hf-repo, e.g. Qwen2-7B-Instruct.Q5_K_M.gguf.
   --hf-mmproj-file value         Multimodal projector file below the --hf-repo.
   --hf-repo value, --hfr value   Repository of HuggingFace which the GGUF file store for the main model, e.g. QuantFactory/Qwen2-7B-Instruct-GGUF, works with --hf-file.
   --hf-token value, --hft value  User access token of HuggingFace, optional, works with --hf-repo/--hf-file pair or --hf-draft-repo/--hf-draft-file pair. See https://huggingface.co/settings/tokens.

   Model/Remote/ModelScope

   --ms-draft-file value   Model file below the --ms-draft-repo, optional, e.g. qwen1_5-1_8b-chat-q5_k_m.gguf.
   --ms-draft-repo value   Repository of ModelScope which the GGUF file store for the draft model, optional, e.g. qwen/Qwen1.5-1.8B-Chat-GGUF, works with --ms-draft-file.
   --ms-file value         Model file below the --ms-repo, e.g. qwen1_5-7b-chat-q5_k_m.gguf.
   --ms-mmproj-file value  Multimodal projector file below the --ms-repo.
   --ms-repo value         Repository of ModelScope which the GGUF file store for the main model, e.g. qwen/Qwen1.5-7B-Chat-GGUF, works with --ms-file.
   --ms-token value        Git access token of ModelScope, optional, works with --ms-repo/--ms-file pair or --ms-draft-repo/--ms-draft-file pair. See https://modelscope.cn/my/myaccesstoken.

   Model/Remote/Ollama

   --ol-base-url value  Model base URL of Ollama, e.g. https://registry.ollama.ai. (default: "https://registry.ollama.ai")
   --ol-model value     Model name of Ollama, e.g. gemma2.
   --ol-usage           Specify respecting the extending layers introduced by Ollama, works with --ol-model, which affects the usage estimation. (default: false)

   Output

   --in-mib             Display the estimated result in table with MiB. (default: false)
   --json               Output as JSON. (default: false)
   --json-pretty        Works with --json, to output pretty format JSON. (default: true)
   --raw                Output the GGUF file information as JSON only, skip anything. (default: false)
   --raw-output value   Works with --raw, to save the result to the file
   --skip-architecture  Skip to display architecture metadata. (default: false)
   --skip-estimate      Skip to estimate. (default: false)
   --skip-model         Skip to display model metadata. (default: false)
   --skip-tokenizer     Skip to display tokenizer metadata. (default: false)

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

+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Causal | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      32768      |     4096      |       4       |       true       |         32         |   32   |      14336       |     0      |     32032      |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | EOT Token | EOM Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama | 450.50 KiB  |   32032    |       N/A        |     1     |   32000   |    N/A    |    N/A    |      N/A      |       N/A       |      N/A      |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+

+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |       UMA (RAM + VRAM)       | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |     2048 / 512     |      false      |     true     |     false      |  33 (32 + 1)   |      Yes       | 88.39 MiB + 4 GiB = 4.09 GiB | 238.39 MiB |  11.06 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------+------------+-------------+

```

#### Parse remote GGUF file

```shell
$ gguf-parser --url="https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/resolve/main/Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf"
+--------------+----------+-------+--------------+---------------+--------+------------+----------+
|      \       |   Name   | Arch  | Quantization | Little Endian |  Size  | Parameters |   BPW    |
+--------------+----------+-------+--------------+---------------+--------+------------+----------+
|    MODEL     | emozilla | llama | Q4_K/Q3_K_M  |     true      | 21 GiB |  46.70 B   | 3.86 bpw |
+--------------+----------+-------+--------------+---------------+--------+------------+----------+

+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Causal | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      32768      |     4096      |       4       |       true       |         32         |   32   |      14336       |     8      |     32002      |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | EOT Token | EOM Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama | 449.91 KiB  |   32002    |       N/A        |     1     |   32000   |    N/A    |    N/A    |       0       |       N/A       |       2       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+

+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |     2048 / 512     |      false      |    false     |     false      |  33 (32 + 1)   |      Yes       | 245.24 MiB + 24.84 GiB = 25.08 GiB | 395.24 MiB |  27.31 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Parse HuggingFace GGUF file

```shell
$ gguf-parser --hf-repo="openbmb/MiniCPM-Llama3-V-2_5-gguf" --hf-file="ggml-model-Q5_K_M.gguf" --hf-mmproj-file="mmproj-model-f16.gguf"
+--------------+-------+-------+----------------+---------------+----------+------------+----------+
|      \       | Name  | Arch  |  Quantization  | Little Endian |   Size   | Parameters |   BPW    |
+--------------+-------+-------+----------------+---------------+----------+------------+----------+
|    MODEL     | model | llama | IQ3_XXS/Q5_K_M |     true      | 5.33 GiB |   8.03 B   | 5.70 bpw |
+--------------+-------+-------+----------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Causal | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      8192       |     4096      |       4       |       true       |         32         |   32   |      14336       |     0      |     128256     |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | EOT Token | EOM Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | gpt2  |    2 MiB    |   128256   |       N/A        |  128000   |  128001   |    N/A    |    N/A    |    128002     |       N/A       |       0       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+

+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+---------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |        UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+---------------------------------+------------+-------------+
|   ESTIMATE   | llama |     8192     |     2048 / 512     |      false      |     true     |     false      |  33 (32 + 1)   |      Yes       | 97.36 MiB + 1.96 GiB = 2.06 GiB | 247.36 MiB |  7.45 GiB   |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+---------------------------------+------------+-------------+

```

#### Parse ModelScope GGUF file

```shell
$ gguf-parser --ms-repo="shaowenchen/chinese-alpaca-2-13b-16k-gguf" --ms-file="chinese-alpaca-2-13b-16k.Q5_K.gguf"
+--------------+------+-------+----------------+---------------+----------+------------+----------+
|      \       | Name | Arch  |  Quantization  | Little Endian |   Size   | Parameters |   BPW    |
+--------------+------+-------+----------------+---------------+----------+------------+----------+
|    MODEL     |  ..  | llama | IQ3_XXS/Q5_K_M |     true      | 8.76 GiB |  13.25 B   | 5.68 bpw |
+--------------+------+-------+----------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Causal | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      16384      |     5120      |       1       |       true       |        N/A         |   40   |      13824       |     0      |     55296      |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | EOT Token | EOM Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama | 769.83 KiB  |   55296    |       N/A        |     1     |     2     |    N/A    |    N/A    |      N/A      |       N/A       |      N/A      |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+

+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+-----------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |         UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+-----------------------------------+------------+-------------+
|   ESTIMATE   | llama |    16384     |     2048 / 512     |      false      |     true     |     false      |  41 (40 + 1)   |      Yes       | 61.18 MiB + 12.50 GiB = 12.56 GiB | 211.18 MiB |  22.74 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+-----------------------------------+------------+-------------+

```

#### Parse Ollama model

```shell
$ gguf-parser --ol-model="gemma2"
+--------------+---------------+--------+--------------+---------------+----------+------------+----------+
|      \       |     Name      |  Arch  | Quantization | Little Endian |   Size   | Parameters |   BPW    |
+--------------+---------------+--------+--------------+---------------+----------+------------+----------+
|    MODEL     | gemma-2-9b-it | gemma2 |     Q4_0     |     true      | 5.06 GiB |   9.24 B   | 4.71 bpw |
+--------------+---------------+--------+--------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Causal | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      8192       |     3584      |       2       |       true       |         16         |   42   |      14336       |     0      |     256000     |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | EOT Token | EOM Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama |  3.80 MiB   |   256000   |       N/A        |     2     |     1     |    N/A    |    N/A    |       3       |       N/A       |       0       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+

+--------------+--------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+---------------------------------+------------+-------------+
|      \       |  Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |        UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+--------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+---------------------------------+------------+-------------+
|   ESTIMATE   | gemma2 |     8192     |     2048 / 512     |      false      |     true     |     false      |  43 (42 + 1)   |      Yes       | 65.97 MiB + 2.62 GiB = 2.69 GiB | 215.97 MiB |  8.43 GiB   |
+--------------+--------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+---------------------------------+------------+-------------+

```

##### Parse Ollama model with its preset params

```shell
$ gguf-parser --ol-model="gemma2" --ol-usage
+--------------+---------------+--------+--------------+---------------+----------+------------+----------+
|      \       |     Name      |  Arch  | Quantization | Little Endian |   Size   | Parameters |   BPW    |
+--------------+---------------+--------+--------------+---------------+----------+------------+----------+
|    MODEL     | gemma-2-9b-it | gemma2 |     Q4_0     |     true      | 5.06 GiB |   9.24 B   | 4.71 bpw |
+--------------+---------------+--------+--------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Causal | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |      8192       |     3584      |       2       |       true       |         16         |   42   |      14336       |     0      |     256000     |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | EOT Token | EOM Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | llama |  3.80 MiB   |   256000   |       N/A        |     2     |     1     |    N/A    |    N/A    |       3       |       N/A       |       0       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+

+--------------+--------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+
|      \       |  Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |         UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+--------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+
|   ESTIMATE   | gemma2 |     2048     |     2048 / 512     |      false      |     true     |     false      |  43 (42 + 1)   |      Yes       | 53.97 MiB + 672 MiB = 725.97 MiB | 203.97 MiB |  6.46 GiB   |
+--------------+--------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+

```

#### Parse Embedding model

```shell
$ gguf-parser --hf-repo="CompendiumLabs/bge-large-zh-v1.5-gguf" --hf-file="bge-large-zh-v1.5-f16.gguf"
+--------------+-------------------+------+--------------+---------------+------------+------------+-----------+
|      \       |       Name        | Arch | Quantization | Little Endian |    Size    | Parameters |    BPW    |
+--------------+-------------------+------+--------------+---------------+------------+------------+-----------+
|    MODEL     | bge-large-zh-v1.5 | bert |     F16      |     true      | 619.50 MiB |  324.47 M  | 16.02 bpw |
+--------------+-------------------+------+--------------+---------------+------------+------------+-----------+

+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
|      \       | Max Context Len | Embedding Len | Embedding GQA | Attention Causal | Attention Head Cnt | Layers | Feed Forward Len | Expert Cnt | Vocabulary Len |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE |       512       |     1024      |       1       |      false       |        N/A         |   24   |       4096       |     0      |     21128      |
+--------------+-----------------+---------------+---------------+------------------+--------------------+--------+------------------+------------+----------------+

+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|      \       | Model | Tokens Size | Tokens Len | Added Tokens Len | BOS Token | EOS Token | EOT Token | EOM Token | Unknown Token | Separator Token | Padding Token |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+
|  TOKENIZER   | bert  | 266.05 KiB  |   21128    |       N/A        |     0     |     2     |    N/A    |    N/A    |      100      |       N/A       |       0       |
+--------------+-------+-------------+------------+------------------+-----------+-----------+-----------+-----------+---------------+-----------------+---------------+

+--------------+------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+--------------------------------+------------+-------------+
|      \       | Arch | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |        UMA (RAM + VRAM)        | NonUMA RAM | NonUMA VRAM |
+--------------+------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+--------------------------------+------------+-------------+
|   ESTIMATE   | bert |     512      |    2048 / 2048     |      false      |     true     |      true      |  25 (24 + 1)   |      Yes       | 18.13 MiB + 48 MiB = 66.13 MiB | 168.13 MiB | 918.48 MiB  |
+--------------+------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+--------------------------------+------------+-------------+

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
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |     2048 / 512     |      false      |    false     |     false      |  33 (32 + 1)   |      Yes       | 245.24 MiB + 24.84 GiB = 25.08 GiB | 395.24 MiB |  27.31 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Estimate with zero layers offload

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --gpu-layers=0
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+-----------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |      UMA (RAM + VRAM)       | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+-----------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |     2048 / 512     |      false      |    false     |     false      |       0        |       No       | 25.09 GiB + 0 B = 25.09 GiB | 25.24 GiB  |  2.46 GiB   |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+-----------------------------+------------+-------------+

```

#### Estimate with specific layers offload

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --gpu-layers=10
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |         UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |     2048 / 512     |      false      |    false     |     false      |       10       |       No       | 17.36 GiB + 7.73 GiB = 25.09 GiB | 17.51 GiB  |  10.19 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+

```

#### Estimate with specific context size

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --ctx-size=4096
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |     4096     |     2048 / 512     |      false      |    false     |     false      |  33 (32 + 1)   |      Yes       | 189.24 MiB + 21.34 GiB = 21.53 GiB | 339.24 MiB |  21.89 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Estimate with Flash Attention

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --flash-attention
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |     2048 / 512     |      true       |    false     |     false      |  33 (32 + 1)   |      Yes       | 245.24 MiB + 24.84 GiB = 25.08 GiB | 395.24 MiB |  25.33 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+

```

#### Estimate with No MMap

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --gpu-layers=10 --no-mmap
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |         UMA (RAM + VRAM)         | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |     2048 / 512     |      false      |    false     |     false      |       10       |       No       | 17.36 GiB + 7.73 GiB = 25.09 GiB | 17.51 GiB  |  10.19 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+----------------------------------+------------+-------------+

```

#### Estimate step-by-step offload layers

```shell
$ gguf-parser --hf-repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --hf-file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --gpu-layers-step=5
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|      \       | Arch  | Context Size | Batch Size (L / P) | Flash Attention | MMap Support | Embedding Only | Offload Layers | Full Offloaded |          UMA (RAM + VRAM)          | NonUMA RAM | NonUMA VRAM |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+
|   ESTIMATE   | llama |    32768     |     2048 / 512     |      false      |    false     |     false      |       0        |       No       |    25.09 GiB + 0 B = 25.09 GiB     | 25.24 GiB  |  2.46 GiB   |
+              +       +              +                    +                 +              +----------------+----------------+                +------------------------------------+------------+-------------+
|              |       |              |                    |                 |              |     false      |       5        |                |  21.23 GiB + 3.86 GiB = 25.09 GiB  | 21.37 GiB  |  6.33 GiB   |
+              +       +              +                    +                 +              +----------------+----------------+                +------------------------------------+------------+-------------+
|              |       |              |                    |                 |              |     false      |       10       |                |  17.36 GiB + 7.73 GiB = 25.09 GiB  | 17.51 GiB  |  10.19 GiB  |
+              +       +              +                    +                 +              +----------------+----------------+                +------------------------------------+------------+-------------+
|              |       |              |                    |                 |              |     false      |       15       |                | 13.50 GiB + 11.59 GiB = 25.09 GiB  | 13.64 GiB  |  14.06 GiB  |
+              +       +              +                    +                 +              +----------------+----------------+                +------------------------------------+------------+-------------+
|              |       |              |                    |                 |              |     false      |       20       |                |  9.63 GiB + 15.46 GiB = 25.09 GiB  |  9.78 GiB  |  17.92 GiB  |
+              +       +              +                    +                 +              +----------------+----------------+                +------------------------------------+------------+-------------+
|              |       |              |                    |                 |              |     false      |       25       |                |  5.77 GiB + 19.32 GiB = 25.09 GiB  |  5.91 GiB  |  21.79 GiB  |
+              +       +              +                    +                 +              +----------------+----------------+                +------------------------------------+------------+-------------+
|              |       |              |                    |                 |              |     false      |       30       |                |  1.90 GiB + 23.19 GiB = 25.09 GiB  |  2.05 GiB  |  25.65 GiB  |
+              +       +              +                    +                 +              +----------------+----------------+----------------+------------------------------------+------------+-------------+
|              |       |              |                    |                 |              |     false      |  33 (32 + 1)   |      Yes       | 245.24 MiB + 24.84 GiB = 25.08 GiB | 395.24 MiB |  27.31 GiB  |
+--------------+-------+--------------+--------------------+-----------------+--------------+----------------+----------------+----------------+------------------------------------+------------+-------------+

```

## License

MIT
