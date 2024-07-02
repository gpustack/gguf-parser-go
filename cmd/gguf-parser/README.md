# GGUF Parser

Review/Check/Estimate [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) file.

## Usage

```shell
$ gguf-parser --help
Usage of gguf-parser ...:
  -ubatch-size int
        Specify the physical maximum batch size, which is used to estimate the usage, default is 512. (default 512)
  -ctx-size int
        Specify the size of prompt context, which is used to estimate the usage, default is equal to the model's maximum context size. (default -1)
  -debug
        Enable debugging, verbosity.
  -file string
        Model file below the --repo, e.g. Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.
  -flash-attention
        Specify enabling Flash Attention, which is used to estimate the usage. Flash Attention can reduce the usage of RAM/VRAM.
  -json
        Output as JSON,
  -json-pretty
        Output as pretty JSON. (default true)
  -kv-type string
        Specify the type of Key-Value cache, which is used to estimate the usage, select from [f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1], default is f16. Use quantization type means enabling --flash-attention as well. (default "f16")
  -no-mmap
        Specify disabling Memory-Mapped using, which is used to estimate the usage. Memory-Mapped can avoid loading the entire model weights into RAM.
  -offload-layers int
        Specify how many layers to offload, which is used to estimate the usage, default is full offloaded. (default -1)
  -offload-layers-step uint
        Specify the step of layers to offload, works with --offload-layers.
  -parallel-size int
        Specify the number of parallel sequences to decode, which is used to estimate the usage, default is 1. (default 1)
  -path string
        Path where the GGUF file to load, e.g. ~/.cache/lm-studio/models/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf.
  -repo string
        Repository of HuggingFace which the GGUF file store, e.g. NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF, works with --file.
  -skip-architecture
        Skip to display architecture metadata.
  -skip-estimate
        Skip to estimate.
  -skip-model
        Skip to display model metadata.
  -skip-tls-verify
        Skip TLS verification, works with --url.
  -skip-tokenizer
        Skip to display tokenizer metadata
  -url string
        Url where the GGUF file to load, e.g. https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf. Note that gguf-parser does not need to download the entire GGUF file.
  -version
        Show gguf-parser version.
```

### Parse

#### parse local GGUF file

```shell
$ gguf-parser --path="~/.cache/lm-studio/models/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF/Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf"
+-------+-------+-------+----------------------+----------------+---------------+----------+------------+----------+
| MODEL | NAME  | ARCH  | QUANTIZATION VERSION |   FILE TYPE    | LITTLE ENDIAN |   SIZE   | PARAMETERS |   BPW    |
+       +-------+-------+----------------------+----------------+---------------+----------+------------+----------+
|       | jeffq | llama |          2           | IQ3_XXS/Q5_K_M |     true      | 4.78 GiB |   7.24 B   | 5.67 bpw |
+-------+-------+-------+----------------------+----------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE | MAX CONTEXT LEN | EMBEDDING LEN | EMBEDDING GQA | ATTENTION HEAD CNT | LAYERS | FEED FORWARD LEN | EXPERT CNT | VOCABULARY LEN |
+              +-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|              |      32768      |     4096      |     1024      |         32         |   32   |      14336       |     0      |     32032      |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+-----------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
| TOKENIZER | MODEL | TOKENS SIZE | TOKENS LEN | ADDED TOKENS LEN | BOS TOKEN | EOS TOKEN | UNKNOWN TOKEN | SEPARATOR TOKEN | PADDING TOKEN |
+           +-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|           | llama | 450.50 KiB  |   32032    |        0         |     1     |   32000   |      N/A      |       N/A       |      N/A      |
+-----------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+----------+-------+--------------+-----------------+--------------+----------------+----------------+----------+------------+-------------+
| ESTIMATE | ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED | UMA RAM  | NONUMA RAM | NONUMA VRAM |
+          +-------+--------------+-----------------+--------------+----------------+----------------+----------+------------+-------------+
|          | llama |    32768     |      false      |     true     |  33 (32 + 1)   |      Yes       | 4.09 GiB | 238.39 MiB |  10.80 GiB  |
+----------+-------+--------------+-----------------+--------------+----------------+----------------+----------+------------+-------------+

```

#### parse remote GGUF file

```shell
$ gguf-parser --url="https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/resolve/main/Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf"
+-------+----------+-------+----------------------+-------------+---------------+--------+------------+----------+
| MODEL |   NAME   | ARCH  | QUANTIZATION VERSION |  FILE TYPE  | LITTLE ENDIAN |  SIZE  | PARAMETERS |   BPW    |
+       +----------+-------+----------------------+-------------+---------------+--------+------------+----------+
|       | emozilla | llama |          2           | Q4_K/Q3_K_M |     true      | 21 GiB |  46.70 B   | 3.86 bpw |
+-------+----------+-------+----------------------+-------------+---------------+--------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE | MAX CONTEXT LEN | EMBEDDING LEN | EMBEDDING GQA | ATTENTION HEAD CNT | LAYERS | FEED FORWARD LEN | EXPERT CNT | VOCABULARY LEN |
+              +-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|              |      32768      |     4096      |     1024      |         32         |   32   |      14336       |     8      |     32002      |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+-----------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
| TOKENIZER | MODEL | TOKENS SIZE | TOKENS LEN | ADDED TOKENS LEN | BOS TOKEN | EOS TOKEN | UNKNOWN TOKEN | SEPARATOR TOKEN | PADDING TOKEN |
+           +-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|           | llama | 449.91 KiB  |   32002    |        0         |     1     |   32000   |       0       |       N/A       |       2       |
+-----------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
| ESTIMATE | ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED |  UMA RAM  | NONUMA RAM | NONUMA VRAM |
+          +-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
|          | llama |    32768     |      false      |    false     |  33 (32 + 1)   |      Yes       | 25.08 GiB | 292.68 MiB |  27.04 GiB  |
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+

```

#### Parse HuggingFace GGUF file

```shell
$ gguf-parser --repo="openbmb/MiniCPM-Llama3-V-2_5-gguf" --file="ggml-model-Q5_K_M.gguf" 
+-------+-------+-------+----------------------+----------------+---------------+----------+------------+----------+
| MODEL | NAME  | ARCH  | QUANTIZATION VERSION |   FILE TYPE    | LITTLE ENDIAN |   SIZE   | PARAMETERS |   BPW    |
+       +-------+-------+----------------------+----------------+---------------+----------+------------+----------+
|       | model | llama |          2           | IQ3_XXS/Q5_K_M |     true      | 5.33 GiB |   8.03 B   | 5.70 bpw |
+-------+-------+-------+----------------------+----------------+---------------+----------+------------+----------+

+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
| ARCHITECTURE | MAX CONTEXT LEN | EMBEDDING LEN | EMBEDDING GQA | ATTENTION HEAD CNT | LAYERS | FEED FORWARD LEN | EXPERT CNT | VOCABULARY LEN |
+              +-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+
|              |      8192       |     4096      |     1024      |         32         |   32   |      14336       |     0      |     128256     |
+--------------+-----------------+---------------+---------------+--------------------+--------+------------------+------------+----------------+

+-----------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
| TOKENIZER | MODEL | TOKENS SIZE | TOKENS LEN | ADDED TOKENS LEN | BOS TOKEN | EOS TOKEN | UNKNOWN TOKEN | SEPARATOR TOKEN | PADDING TOKEN |
+           +-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+
|           | gpt2  |    2 MiB    |   128256   |        0         |  128000   |  128001   |    128002     |       N/A       |       0       |
+-----------+-------+-------------+------------+------------------+-----------+-----------+---------------+-----------------+---------------+

+----------+-------+--------------+-----------------+--------------+----------------+----------------+----------+------------+-------------+
| ESTIMATE | ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED | UMA RAM  | NONUMA RAM | NONUMA VRAM |
+          +-------+--------------+-----------------+--------------+----------------+----------------+----------+------------+-------------+
|          | llama |     8192     |      false      |     true     |  33 (32 + 1)   |      Yes       | 1.08 GiB | 234.61 MiB |  6.55 GiB   |
+----------+-------+--------------+-----------------+--------------+----------------+----------------+----------+------------+-------------+

```

### Estimate

#### Estimate with zero layers offload

```shell
$ gguf-parser --repo="mradermacher/Falcon2-8B-Dutch-GGUF" --file="Falcon2-8B-Dutch.Q5_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --offload-layers=0
+----------+--------+--------------+-----------------+--------------+----------------+----------------+------------+------------+-------------+
| ESTIMATE |  ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED |  UMA RAM   | NONUMA RAM | NONUMA VRAM |
+          +--------+--------------+-----------------+--------------+----------------+----------------+------------+------------+-------------+
|          | falcon |     2048     |      false      |     true     |       0        |       No       | 383.46 MiB | 533.46 MiB | 404.91 MiB  |
+----------+--------+--------------+-----------------+--------------+----------------+----------------+------------+------------+-------------+

```

#### Estimate with specific layers offload

```shell
$ gguf-parser --repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --offload-layers=10
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
| ESTIMATE | ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED |  UMA RAM  | NONUMA RAM | NONUMA VRAM |
+          +-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
|          | llama |    32768     |      false      |    false     |       10       |       No       | 25.08 GiB | 17.50 GiB  |  9.83 GiB   |
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+

```

#### Estimate with specific context size

```shell
$ gguf-parser --repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --ctx-size=4096
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
| ESTIMATE | ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED |  UMA RAM  | NONUMA RAM | NONUMA VRAM |
+          +-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
|          | llama |     4096     |      false      |    false     |  33 (32 + 1)   |      Yes       | 21.53 GiB | 236.68 MiB |  21.74 GiB  |
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+

```

#### Estimate with Flash Attention

```shell
$ gguf-parser --repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --flash-attention
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
| ESTIMATE | ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED |  UMA RAM  | NONUMA RAM | NONUMA VRAM |
+          +-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
|          | llama |    32768     |      true       |    false     |  33 (32 + 1)   |      Yes       | 25.08 GiB | 292.68 MiB |  25.18 GiB  |
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+

```

#### Estimate with No MMap

```shell
$ gguf-parser --repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --offload-layers=10 --no-mmap
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
| ESTIMATE | ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED |  UMA RAM  | NONUMA RAM | NONUMA VRAM |
+          +-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
|          | llama |    32768     |      false      |    false     |       10       |       No       | 25.08 GiB | 17.50 GiB  |  9.83 GiB   |
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+

```

#### Estimate step-by-step offload layers

```shell
$ gguf-parser --repo="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF" --file="Nous-Hermes-2-Mixtral-8x7B-DPO.Q3_K_M.gguf" --skip-model --skip-architecture --skip-tokenizer --offload-layers-step=5
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
| ESTIMATE | ARCH  | CONTEXT SIZE | FLASH ATTENTION | MMAP SUPPORT | OFFLOAD LAYERS | FULL OFFLOADED |  UMA RAM  | NONUMA RAM | NONUMA VRAM |
+          +-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+
|          | llama |    32768     |      false      |    false     |       0        |       No       | 25.08 GiB | 25.23 GiB  |  2.10 GiB   |
+          +       +              +                 +              +----------------+                +           +------------+-------------+
|          |       |              |                 |              |       5        |                |           | 21.36 GiB  |  5.97 GiB   |
+          +       +              +                 +              +----------------+                +           +------------+-------------+
|          |       |              |                 |              |       10       |                |           | 17.50 GiB  |  9.83 GiB   |
+          +       +              +                 +              +----------------+                +           +------------+-------------+
|          |       |              |                 |              |       15       |                |           | 13.63 GiB  |  13.70 GiB  |
+          +       +              +                 +              +----------------+                +           +------------+-------------+
|          |       |              |                 |              |       20       |                |           |  9.77 GiB  |  17.56 GiB  |
+          +       +              +                 +              +----------------+                +           +------------+-------------+
|          |       |              |                 |              |       25       |                |           |  5.91 GiB  |  21.42 GiB  |
+          +       +              +                 +              +----------------+                +           +------------+-------------+
|          |       |              |                 |              |       30       |                |           |  2.04 GiB  |  25.29 GiB  |
+          +       +              +                 +              +----------------+----------------+           +------------+-------------+
|          |       |              |                 |              |  33 (32 + 1)   |      Yes       |           | 292.68 MiB |  27.04 GiB  |
+----------+-------+--------------+-----------------+--------------+----------------+----------------+-----------+------------+-------------+

```

## License

MIT
