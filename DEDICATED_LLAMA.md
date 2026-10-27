# Branch Context Benchmark v1.1 — fresh llama.cpp process per trial

**This option does not use Jan's router and does not need an internal API key.**
The original Jan MCP extension is still included. This is an alternative inference
engine for the benchmark only, using the same branching, summarizing and linear scenarios.

## 1. Install llama.cpp on Arch Linux (AMD GPU)

```bash
sudo pacman -S llama-cpp ggml-hip
command -v llama-server
```

`ggml-hip` is the ROCm GPU backend. The package is separate from Jan's Flatpak worker.
Make sure the GPU has sufficient free VRAM; if Jan has the same model already
loaded, unload it from Jan's UI before running these tests. The benchmark does
not unload Jan's model.

## 2. Locate your already-downloaded Qwen GGUF file

```bash
find "$HOME/.var/app/ai.jan.Jan" "$HOME/Documents" -type f -iname '*Qwen*gguf' 2>/dev/null
```

If the model is on a separate SSD or a custom Jan storage location, search that
location instead. **Use the real `.gguf` file**, not the model name Jan displays.

Assign its path (quotes preserve spaces):

```bash
MODEL='/absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf'
```

## 3. Read-only preflight (does not start anything)

```bash
python3 benchmark.py --engine llama --llama-model "$MODEL" --check-llama-server
```

## 4. Small live smoke test (starts and stops its own llama-server)

```bash
python3 benchmark.py --engine llama --llama-model "$MODEL" --scenario repair --smoke --context-limit 8192
```

## 5. Longer run (three competitors)

```bash
python3 benchmark.py --engine llama --llama-model "$MODEL" \
  --scenario repair --tasks 1 2 4 --repeats 3 \
  --nested-depth 2 --pressure-records 4 --pressure-detail-chars 48 \
  --max-actions 32 --context-limit 8192
```

**Default behavior:** the benchmark starts a brand-new `llama-server` process
BEFORE EACH trial, waits for `/health` and `/v1/models`, measures the trial, then
stops the server process it started. The restart time is reported separately.
The results still go to `data/benchmarks/` inside the project. Logs from
server startup are placed in that run's `server_logs/` subdirectory.

Optional flags:

- `--llama-server /path/to/llama-server` chooses a binary if it is not in PATH.
- `--llama-port 18090` chooses another dedicated loopback port if 18089 is occupied.
- `--llama-gpu-layers 99` controls GPU offloading; 0 for CPU (very slow).
- `--llama-startup-timeout 300` allows more model loading time.
- `--llama-arg=--some-supported-option` passes a single option through to llama-server.
- `--reuse-llama` starts the server once and does not clear its cache between trials.

The server listens **only on 127.0.0.1**, does not enable any server-side
shell/tool execution capabilities, and needs no Jan authentication. If port
18089 is occupied, it stops safely without touching other processes.
On interruption or startup failure, only the subprocess group launched by
this benchmark is terminated.

If the GPU fails to initialize or the model is unsupported, the benchmark
reports a startup error and points you to `server_logs/` for the actual llama.cpp
message. No guessed fallback to CPU is performed by the benchmark (the chosen
llama.cpp binary and backend control the actual offload).

### What was tested

Offline integration tests use an independent **fake** HTTP inference server.
They verify process creation, unique PIDs for successive trials, health checks,
refusal to take over an occupied port, failure cleanup, and all three benchmark
modes. **They do not prove real Qwen / AMD ROCm model loading on Build-Arch.**

### Comparison caveat

The same Qwen GGUF can be used, but Jan Desktop's bundled llama.cpp version,
chat template, GPU settings and inference sampler can differ from the standalone
Arch package. Therefore compare all three competitors **within the same new
benchmark run**; don't assume wall-clock or token parity with older Jan results.
