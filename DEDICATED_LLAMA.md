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

## If a repair returns HTTP 400 near the context limit

The dedicated server may reject a chat request once the **input history plus
reserved output** no longer fits within `--context-limit`. A repair with default
nested depth 2 and pressure records 4 can overwhelm an 8K context **even in
branch mode**, because branches are isolated *per repair*, not per nested step.
This is not a failure to start llama-server, and it does not imply Jan API auth.

Start with a deliberately lighter 8K baseline to verify inference/actions:

```bash
python3 benchmark.py --engine llama --llama-model "$MODEL" \
  --scenario repair --smoke --context-limit 8192 \
  --nested-depth 0 --pressure-records 0 \
  --recovery-log-lines 8 --max-actions 20
```

For the **full nested-depth-2 test**, use a larger server context if your GPU
has enough available memory (for example, start with 16384, then 32768 if
necessary), or lower the evidence pressure. Larger contexts consume more memory.

`--dry-run` prints a rough warning when mandatory evidence plus output reserve
approaches the context budget. On HTTP errors, the benchmark now provides a
sanitized cause if llama.cpp explicitly reports context overflow. For other
HTTP 400 errors, consult the actual response in the `server_logs/` directory.

## v1.2: final-report reliability and the first real multi-repair comparison

A Qwen3.5 thinking model may use its entire output allowance for reasoning on a simple final report. The benchmark now gives final reports a separate **2048-token** allowance (investigation actions stay at 1024), requests `reasoning_effort=none` for simple acknowledgement/report calls only, and retries one time if the model returns an empty/truncated final report. This per-request hint is not guaranteed to work on every llama.cpp version. Every attempt is measured; strict pass still requires a correct model-produced JSON report.

**First, verify the revised final-report behavior:**

```bash
python3 benchmark.py --engine llama --llama-model "$MODEL" \
  --scenario repair --smoke --nested-depth 0 --pressure-records 0 \
  --recovery-log-lines 8 --context-limit 8192
```

**Then test three incidents per mode** (summarized actually compacts after two):

```bash
python3 benchmark.py --engine llama --llama-model "$MODEL" \
  --scenario repair --tasks 3 --repeats 1 --nested-depth 0 \
  --pressure-records 0 --recovery-log-lines 8 --context-limit 16384
```

If your build rejects the per-request reasoning setting, add `--report-reasoning auto`. You may explicitly choose the output cap with `--final-output-tokens 2048` or disable retries with `--final-retries 0`. Increasing the final cap consumes some of the same context capacity, so reduce pressure or increase context cautiously. Check `results.json` → `trials[].calls[]` for `visible_output_chars`, `reasoning_output_chars`, `reasoning_tokens`, `requested_reasoning_effort` and `finish_reason`. Thinking *can* be hidden even when the API does not report separate reasoning tokens.

The additional `compiled_report` field is assembled by the benchmark after verifying repairs in its simulator. It is **not** scored as a successful *model* final report; it simply lets you see whether repair work was completed even when the AI's JSON failed.

## Periodic summary controls (v1.3)

When comparing all three context methods, summarization is performed only
after the configured number of completed repairs (default two) and not after
the final repair. A three-repair run exercises it:

```bash
python3 benchmark.py --engine llama --llama-model "$MODEL" \
  --scenario repair --tasks 3 --repeats 1 \
  --nested-depth 0 --pressure-records 0 --recovery-log-lines 8 \
  --context-limit 16384 --summary-output-tokens 2048 --summary-retries 1
```

The summary generation cap is distinct from both the investigation cap
(default 1024) and the retained summary memory budget (1800 characters).
Summary calls use the reporting no-reasoning hint when enabled; the GPU and
model decide whether that hint is honored. A summary that remains empty or
cut off after its retry fails explicitly; it is never filled from the
benchmark answer key. All calls are measured and recorded.

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
