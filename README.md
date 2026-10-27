# New in v1.2: reliable reporting diagnostics and multi-repair comparison

The dedicated llama.cpp benchmark now distinguishes **investigation** from **reporting** calls. By default, the investigation still uses normal model reasoning, while ACK and final-report calls request `reasoning_effort=none` (best effort; some older backends ignore the request). Final reports have a separate **2,048-token** cap and get one retry if output is empty or truncated. All requests, including retries, count toward runtime and token totals.

`results.json`, `results.csv`, and `SUMMARY.md` separately expose empty visible responses, optional reported reasoning-token counts, first final-report outcome, final attempts, and retry recovery. The repair scenario also records an *orchestrator-compiled* report derived only from simulated VERIFIED outcomes. **That is not a model-generated report and never counts as strict final success.**

```bash
# One easy incident across all three methods
python3 benchmark.py --engine llama --llama-model "$MODEL" \
  --scenario repair --smoke --nested-depth 0 --pressure-records 0 \
  --recovery-log-lines 8 --context-limit 8192

# A meaningful comparison: 3 repairs per mode, summarized mode compacts after 2
python3 benchmark.py --engine llama --llama-model "$MODEL" \
  --scenario repair --tasks 3 --repeats 1 --nested-depth 0 \
  --pressure-records 0 --recovery-log-lines 8 --context-limit 16384
```

`--report-reasoning auto` disables the per-request no-thinking hint if your llama.cpp build rejects or ignores it. `--final-output-tokens 2048` and `--final-retries 1` are configurable. Backend support for per-request reasoning is version/model dependent; the benchmark logs what it *requested*, not proof of compliance.

---

# New in v1.1: dedicated llama.cpp benchmark engine

**Recommended for clean benchmarking:** use `--engine llama` to run a separate
`llama-server` process with your local GGUF model. It restarts between every
trial without attempting to authenticate to Jan's internal router. It does not
stop or modify Jan. See [DEDICATED_LLAMA.md](DEDICATED_LLAMA.md) for installation
and commands for Arch Linux / AMD ROCm.

The Jan MCP extension and old `--engine jan` path remain available, but the
Jan-specific router reset is not needed for this new mode.

---

## v1.0.4 router authentication diagnosis

`jan_reset.py` now tries the benchmark's existing public Jan API key as a candidate
for internal router authentication. The key is never printed or included in
results. Jan may use a different private key or have an upstream authentication
bug, so a 401 can still be unavoidable. Automatic resets are disabled unless
preflight verifies the router safely. See BENCHMARK.md.

---

**v1.0.3 authentication update:** On Linux Flatpak, Jan normally stores its logs in `~/.var/app/ai.jan.Jan/data/Jan/data/logs/app.log`. The router detector now checks that location in addition to the native/custom data location. It prioritizes an explicit `JAN_ROUTER_API_KEY`, accepts more worker launch-argument formats, and reports *no key found* versus *candidate keys rejected*, without showing secrets. Only an exact Jan-owned port is probed and `--check-jan-reset` remains read-only. **Live validation on Build-Arch is still required.**

# Branch Context for Jan — v1.0.2 (authenticated router reset)

**A working, dependency-free Python MCP server that runs isolated side-task conversations against your local Jan model.** It was designed for Jan Desktop on Linux (including Qwen models), but the MCP server itself is platform independent.

The tool never rewrites the active Jan chat. Instead, the parent model uses an MCP tool to run a **separate context**, and receives only a compact result. A saved branch ID can be reopened later. Nested branches are supported (up to depth 2 by default).

## NEW in v1.0.2: Detect and authenticate to Jan's internal router

This addresses a real failure where Jan's worker socket was correctly found (for example
`ss ports=[41225]`) but the router's `/models` endpoint did not respond as an unauthenticated
llama.cpp API. In some Jan builds the router has its **own private API key**, distinct from the
public Local API Server on port 1337. The earlier preflight treated HTTP 401 like a missing router.

v1.0.2 makes discovery report HTTP errors explicitly and, **only for confirmed Jan-worker ports**, tries private router credentials available in the Jan worker process tree (`--api-key` and `LLAMA_API_KEY`), or in the matching `Router argv` log line in the worker's own Jan data folder. The log fallback requires the **exact port**. Credentials are used only in localhost HTTP requests, are never printed, and are not stored in benchmark reports; system proxy settings are bypassed. Model IDs are also compared with or without a `.gguf` suffix, but only when the match is unambiguous.

**Try first:**

```bash
python3 benchmark.py --check-jan-reset
```

If access is still denied, the error now indicates authentication explicitly. You can obtain
Jan's *internal router* key locally and supply it as `JAN_ROUTER_API_KEY` in the terminal
(rather than putting it in the benchmark config or shell history). For example:

```bash
read -r -s -p 'Jan router API key: ' JAN_ROUTER_API_KEY; echo
export JAN_ROUTER_API_KEY
python3 benchmark.py --check-jan-reset
unset JAN_ROUTER_API_KEY
```

The above optional step requires the **internal router** key, *not* necessarily the API key
in Jan's Local API Server settings. If it still fails, run this read-only command and share
just the HTTP status code (not an API key or raw Jan logs):

```bash
curl --noproxy '*' -s -o /dev/null -w 'Router /models HTTP: %{http_code}\n' http://127.0.0.1:41225/models
```

Replace `41225` with the current Jan-owned port shown by preflight, since it changes between sessions.
**Do not manually kill the worker** to get a new port. The benchmark only performs unload/reload
when a selected model and its router status are both positively verified.

## NEW in v1.0: Clear Jan model memory between trials

v1.0.1 adds a `/proc` fallback when `ss -p` cannot show Jan Flatpak worker ownership. `--check-jan-reset` now checks without generating benchmark scenarios, and explains which Jan-owned ports and model IDs were detected.

The benchmark can now request a **verified unload/reload of the selected model** using Jan's local llama.cpp model router before each test. **Jan Desktop stays running**. This is different from killing the entire `jan-llama-worker` or Flatpak application: killing the worker can also interrupt Jan's API server, so the benchmark does not do that.

A model reset releases the model process and its associated live KV cache. Other processes, the operating system's page cache, and any other Jan models are **not** reset. There is no guarantee that overall RAM usage returns to exactly the same number.

**To use it with your locally running Jan:**

```bash
# First: only check whether the model router is discoverable (NO changes to Jan)
python3 benchmark.py --scenario repair --check-jan-reset

# Small real test: unload/reload the model before all three trials
python3 benchmark.py --scenario repair --smoke --restart-jan-between-trials

# Full comparison: every trial begins with a verified clean model reload
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --nested-depth 2 --pressure-records 4 --pressure-detail-chars 48 \
  --max-actions 32 --context-limit 8192 \
  --restart-jan-between-trials --cooldown-seconds 8
```

The script automatically detects Jan's localhost llama.cpp router by examining the `jan-llama-worker` listening socket on Linux. If this is not possible (e.g. Flatpak/sandbox socket ownership isn't visible), it **stops without terminating Jan**. You may supply `--jan-router-url http://127.0.0.1:PORT` once you have positively identified the actual Jan **model router** port. This is NOT necessarily the standard Jan API port `1337`, and `/v1/models` is not the router's `/models` endpoint. Do not guess a port.

Supported control flags:

| Flag | Effect |
|---|---|
| `--reuse-jan` (default) | Original behavior; caches may persist across trials. |
| `--restart-jan-between-trials` | Unload/reload before **every** trial, including the first. |
| `--restart-jan-between-modes` | Unload/reload before first trial and whenever benchmark mode changes. |
| `--check-jan-reset` | Check model/router visibility without unloading or benchmarking. |
| `--cooldown-seconds 8` | Wait after unload before loading again (default 8 seconds). |
| `--reset-timeout-seconds 180` | Maximum wait per unload or reload transition (default 180 seconds). |
| `--jan-router-url URL` | Override automatic router discovery with verified loopback URL. |

**Safety:** If unload, reload, or Jan API health verification fails, the benchmark **stops** and saves partial results. It does not continue with a potentially dirty trial. The model might remain unloaded after a failed reset; you can use Jan's model controls to load it again. Avoid using Jan for another chat during clean benchmark runs, since that may load models or alter memory conditions.

The outputs now record `reset_before_trial`, `reset_status`, `reset_seconds`, reset method, and Linux `MemAvailable` before/after unload/reload. Reset logs are also written to `results.json`; `SUMMARY.md` reports the reset policy and number of verified resets. These numbers are diagnostics, not proof that memory was fully reclaimed. **Reset overhead is excluded from the per-trial model `elapsed_seconds`.**

Offline tests cover authenticated router discovery, model-ID matching, the reset sequence, report persistence, and original three-way accuracy. A live test on your Jan 0.8.7 installation is still needed.

---

## NEW in v0.9: More context pressure *within* each repair

The Repair Gauntlet now inserts near-matching, structured historical records into **every evidence document that an agent must inspect**: the release registry, approval ledger, alias graph, route inventory, certificate validation, and both shards at each enabled nested index/escrow/recovery level. Unlike optional `logs/diagnostics`, this evidence is on the actual path to solving the incident.

Pressure is configurable and enabled by default on the **command-line benchmark**:

- `--pressure-records 4` (default): extra near-match rows **per required document**. Set to `0` to reproduce the v0.8 evidence volume. Range: 0–160.
- `--pressure-detail-chars 48` (default): extra audit/trace metadata length on each injected record. Range: 0–256.
- `--nested-log-lines` and `--recovery-log-lines` still control the existing obligatory historical shards independently; `--trap-log-lines` still controls **optional** decoy output.

Examples:

```bash
# Safe preflight: see expected required evidence size without connecting to Jan
python3 benchmark.py --scenario repair --tasks 1 2 4 --dry-run \
  --pressure-records 4 --pressure-detail-chars 48

# Smoke test: all three modes, one repair each
python3 benchmark.py --scenario repair --smoke

# Main comparison with moderate pressure
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --nested-depth 2 --pressure-records 4 --pressure-detail-chars 48 \
  --max-actions 32 --context-limit 8192

# A deliberately harder run (do a dry-run first)
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --nested-depth 2 --pressure-records 8 --pressure-detail-chars 64 \
  --max-actions 32 --context-limit 8192
```

**Integrity rules:** All three modes see the exact same seeded incident and extra rows. Pressure records are *not* operative instructions. The correct repair, recovery tokens, and verification seals are unchanged when you vary the pressure settings for the same seed. Pressure-enabled cases also require the agent to inspect all five authoritative sources before a repair can be accepted. Successful repairs still require actual `repair` and `verify` actions.

`results.csv`, `results.json`, and `SUMMARY.md` now include `pressure_generated_chars`, `pressure_evidence_chars` (actually viewed, uniquely per document), `authority_docs_viewed`, `authority_evidence_chars`, and `authority_blocked`. Dry runs estimate mandatory evidence size and warn when required text alone likely exceeds `--context-limit`.

**Important:** The configured 8,192 context budget is not made larger by this feature. A pressure level that overflows a *single* branch can make **all three methods fail** because branching still happens between incidents, not at each nested prerequisite. Use the preflight warning and adjust difficulty gradually. The warning uses characters/4, *not* the model's actual tokenizer; cumulative chat history adds additional pressure.

Offline checks: **47 tests passed**, including three-mode full repair workflows with added pressure, seed-invariant answers, mandatory evidence enforcement, CLI warnings, and prior regressions. These checks use a fake Jan API, not a live Qwen run.

---

## NEW in v0.8: Nested, unavoidable complications

**Each repair now starts with up to two extra prerequisite problems** before the v0.7 authorization recovery can begin. With the default `--nested-depth 2`, the dependency chain is:

```text
Repair route
  └── Approval ledger locked
        └── Recovery evidence locked
              └── Escrow locked
                    └── Evidence index corrupt
                          ├── Inspect index shard A + B
                          └── Rebuild index with reconstructed CURRENT token
                    ├── Inspect escrow shard A + B
                    └── Unlock escrow with reconstructed CURRENT token
              ├── Inspect recovery shard A + B
              └── Recover approval ledger
        └── Inspect approved releases
  └── Select route, APPLY virtual repair, VERIFY seal
```

Every layer has its **own randomized run ID, join order, matching CURRENT records and misleading historical records**. The simulator rejects attempts to read locked shards and will not accept a correct repair until all prerequisites have been solved. The AI cannot skip the investigation by guessing a token: both pieces of evidence must have been inspected.

The same dependency chain is given to `linear`, `summarized`, and `branch`, with unchanged scoring and 1,024 output tokens per model call. Results now expose nested shard reads, index/escrow successes and failures, blocked actions, and compulsory evidence characters separately from optional trap waste.

**Important distinction:** This adds nested *problems*, not independently spawned recursive *context branches*. The benchmark still isolates **each incident** in branch mode, and all nested recovery steps run inside that incident's context. This intentionally tests how much a single isolated branch can handle; testing automatic mid-incident branch creation is a separate next step.

Start with a small comparison:

```bash
python3 benchmark.py --scenario repair --smoke --nested-depth 2 --nested-log-lines 8
```

Run a stronger comparison after confirming Jan's API is stable:

```bash
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --nested-depth 2 --nested-log-lines 14 \
  --recovery-log-lines 36 --trap-density 3 --trap-log-lines 64 \
  --max-actions 32 --context-limit 8192
```

Adjust with `--nested-depth 0` (v0.7-style, no extra stages), `1` (escrow only), or `2` (index + escrow). Increase `--nested-log-lines` to add **unavoidable** context pressure. Each stage adds three minimum decisions (two reads plus one reconstruction); allow at least **18 actions per incident** for the full depth-2 run, preferably 32 or more so the AI can recover from wrong turns. Overly large shard logs may exceed the model's *individual branch* context, which is a legitimate failure to measure.

Offline regression tests check that prerequisites cannot be bypassed and that a scripted evidence-reading agent can complete all three context strategies. These are harness tests, **not performance results from your local Qwen model**.

## Previous change: Unavoidable recovery investigations (v0.7)

There are now **two benchmark scenarios**, each comparing three strategies using the same Jan model:

| Strategy | Context policy |
|---|---|
| `linear` | Keep all investigative steps and noisy simulator outputs in the main history. |
| `summarized` | Let Jan summarize completed investigations, then drop their detailed history. |
| `branch` | Run each incident in a separate local-model conversation; pass only the verified outcome to the parent. |

The new `--scenario repair` is an **interactive stateful simulator**, not a static question. The model must inspect records, resolve contradictions, select a valid configuration, apply a **virtual** repair, and run `verify`. The simulator intentionally supplies misleading historical notes, old console shortcuts, unusable routes, conflicting release approvals, expired certificates, and noisy logs. A wrong repair generates a substantial log detour instead of giving the solution. Each case varies; the same repair cannot always be reused.

**New in v0.7:** Every incident starts with a LOCKED authorization ledger. The model must inspect two noisy recovery shards, identify the *current* incident's fragments among misleading historical entries, combine the fragments in a randomized order, and submit a `recover` action. Until this is done, the simulator will refuse even an otherwise correct repair. This forces real investigative reading in **all three modes**, independently of whether they choose to inspect optional decoys.

The volume of required noise is controlled separately with `--recovery-log-lines` (default 36 per shard). The results distinguish required evidence from wasted optional detours; they record recovered cases, recovery attempts, failed recovery actions, blocked actions, shard reads, and required evidence characters.

The simulator's oracle is generated independently. The model can only obtain a verification seal by completing the repair correctly and asking the simulator to verify it. The same randomized case is given to all three strategies. **Branching is orchestrated per incident**; this benchmark does not claim that the Jan UI independently selects when to branch, and the virtual simulator is *not* a tool installed into your Jan chat.

### Run on your local Jan server

```bash
# The original lookup benchmark is unchanged:
python3 benchmark.py --scenario archive --smoke

# Quick end-to-end test of the new investigation scenario:
python3 benchmark.py --scenario repair --smoke

# Three-way comparison, moderate difficulty:
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 2 --context-limit 8192

# Stronger *mandatory* recovery investigations plus optional dead-end traps:
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --trap-density 3 --trap-log-lines 64 --recovery-log-lines 48 \
  --max-actions 32 --nested-depth 2 --context-limit 8192
```

The existing **1,024 output tokens per model request** remains the default. `--max-actions 32` caps the number of AI decisions per incident so an agent cannot search indefinitely. The old `--records` parameter only applies to the archive scenario. Results are written inside `data/benchmarks/` as `SUMMARY.md`, `results.csv`, and `results.json`.

The new scenario reports how many cases were **verified**, actions taken, rejected repairs, misleading documents opened, *mandatory recovery evidence consumed*, failed recovery attempts, text consumed by optional detours, exact final correctness, total tokens, peak context, and wall-clock time. It is designed to make context pollution observable without granting the model any real execution privileges.

Offline validation: **47 automated tests pass**, including a scripted-agent recovery from a deliberately wrong repair. These tests verify the *benchmark harness*, **not** that branching wins with a real Jan model. Run the commands above to collect model measurements. See [`BENCHMARK.md`](BENCHMARK.md) for details.

The MCP server implementation is unchanged. Keep your existing project-local `config.json` when upgrading; it is deliberately excluded from the downloadable ZIP.

## What this version does

- `branch_solve`: creates a new isolated side task using Jan's OpenAI-compatible Local API Server.
- `branch_resume`: continues an earlier branch, from saved result or an optional full transcript.
- `branch_get`: fetches a branch result (or its optional transcript).
- `branch_list`: lists saved branches.
- `branch_health`: tests the Jan API connection and reports the chosen model.
- A branch can **delegate a smaller branch** to the same Jan model via the `solve_nested_branch` model tool.
- Optional safe *read-only* project file search and reading inside a specifically configured directory.
- All application configuration and branch data live in this project folder: `config.json` and `data/branches/`.
- Only the local Jan HTTP API (`localhost`) is permitted. No Python package installs or cloud account are required.

## Limitations

1. The main Jan conversation **is not actually rewound**, and ordinary Jan tool-call records still take some tokens. This works by isolating the side-task conversation before it occurs. Rewriting Jan's chat-history engine would require modifying Jan itself.
2. A branch can reason, read/search permitted project files, and start nested branches. **It does not inherit Jan's existing Browser MCP, Playwright MCP, external tools, shell, or website session**. Parent Jan must do any browser actions itself. This version is not an autonomous browser-operating side agent.
3. Each branch calls the same local model through Jan's API. This trades extra inference time for a smaller parent context; it does **not** expand your model's real context/VRAM.
4. The parent model needs good MCP tool-calling support. A model can fail to call the tools even when Jan advertises them.
5. The local API integration has been exercised against a mocked Jan-compatible API; **the actual Jan Desktop/Flatpak connection has not been live-tested on your hardware**.

## Setup on your Arch Linux PC

### 1. Extract the package

Put this folder wherever you keep projects, for example:

```bash
mkdir -p /home/andr8076/Documents/Github
# Extract jan-branch-context into /home/andr8076/Documents/Github/
cd /home/andr8076/Documents/Github/jan-branch-context
```

Do not move the `branch_context.py` file outside this folder: all local data paths are relative to it.

### 2. Start Jan's Local API Server

In Jan Desktop:

1. Open **Settings → Local API Server**.
2. Select/load your local model (for example your Qwen3.5-9B) under Default Model, then **Start Server**.
3. Use **127.0.0.1** as host, **1337** as port, and **/v1** as the API prefix, unless you edit the matching values in `config.json`.
4. In the API Server advanced settings, leave **Execute Tools on Server OFF** so this extension handles its own branch tool calls.
5. If you configured an API key in Jan, put it under `api_key` in `config.json` (or pass `JAN_BRANCH_API_KEY` as a process environment variable).
6. With local AI, confirm the selected model is actually **local**, not a remote Jan provider.

Official Jan docs: https://docs.jan.ai/desktop/api-server

**Important:** Your server must stay running while branches are in use. If Jan's local API is crashing, solve that separately before expecting branching to work.

### 3. Configure this extension

Open `config.json` in this folder:

```json
{
  "api_base_url": "http://127.0.0.1:1337/v1",
  "api_key": "",
  "model": "auto",
  "request_timeout_seconds": 180,
  "max_output_tokens": 700,
  "max_agent_turns": 5,
  "max_branch_depth": 2,
  "max_parent_result_chars": 1800,
  "max_context_chars": 5500,
  "save_transcripts": false,
  "read_only_project_root": ""
}
```

- **model**: `auto` picks the first model returned by Jan `/v1/models`; if that is wrong or unavailable, specify the **exact model ID** from Jan's `/v1/models` response.
- **save_transcripts**: `false` (recommended) persists compact outcomes only; on `branch_resume` the model reconstructs context from the previous result. Set `true` to preserve more detailed conversation history on disk, if you understand the privacy implications.
- **read_only_project_root**: an optional absolute folder path. Leave blank to disable file access for the child. Never set it to a directory containing credentials.
- Use `chmod 600 config.json` on Linux after editing if it contains a key.

Now verify connectivity from a terminal:

```bash
python3 branch_context.py doctor
```

Expected result: a JSON object with `"ok": true` and an actual model ID. Try a real test task:

```bash
python3 branch_context.py demo
```

### 4. Connect it as an MCP server in Jan Desktop

Go to **Settings → MCP Servers → Add**. Use:

| Field | Value |
|---|---|
| Server Name | `branch-context` |
| Command | `/usr/bin/python3` |
| Arguments | `/home/andr8076/Documents/Github/jan-branch-context/branch_context.py` |
| Environment Variables | none, if configured via `config.json` |

**Change the argument path if you extracted to another directory.** Jan's setup screen may require adding one argument as a separate field.

After connecting:

- Set **MCP tool-call timeout** in Jan Settings from its default 30 seconds to **300–600 seconds**, if supported. A local reasoning branch may take longer than 30 seconds.
- Ensure the active model advertises and supports **tool calling** in Jan's Model Capabilities.
- Check that `branch_health`, `branch_solve`, `branch_resume`, `branch_get`, and `branch_list` appear.
- Start a fresh chat or reselect your model if tools are not visible.

**Flatpak note:** Jan on Linux may run inside a Flatpak sandbox. If it cannot spawn `/usr/bin/python3` or access the script, use a host execution bridge where permitted (`flatpak-spawn --host /usr/bin/python3 /absolute/path/to/branch_context.py` with the command and arguments split as Jan requires), or grant the Flatpak appropriate access to the specific project directory. Do **not** grant full-home/filesystem access just to make this extension work. Host spawning requires the appropriate Flatpak permission; it may not be available in your installation.

### 5. Tell Jan when to branch

Copy the contents of `JAN_ASSISTANT_INSTRUCTIONS.txt` into your Jan assistant/system instructions. In a new chat, you can test with:

> We are making a cake. We reached the glazing stage, but no glaze is available. Use branch_solve to work out a homemade glaze using common ingredients, then return only the steps and continue the cake plan. Check branch_health first.

Your model should use `branch_solve`, and the MCP response should include a `branch_id`, compact result, and `next_step`. Use `branch_get` or `branch_resume` later.

## What stays local and what does not

The extension sends all generated side-task prompts **only to the configured loopback Jan API**. If that Jan API is configured to route to an external provider, the model request can still leave your computer. Set Jan to a local model for offline use.

Branch metadata (including task strings and results) is stored on disk as JSON files with owner-only permissions on POSIX systems (`0700` for the data folder, `0600` for files). Full conversation transcripts are **off by default**. Do not put passwords, session cookies, API keys, or account identifiers in side-branch tasks. Jan MCP tool-call requests and Jan's own server logs may also retain sensitive arguments; for password work consider disabling verbose API logs and never pass credentials to this tool.

This extension runs **no arbitrary shell commands**, performs **no writes to configured project files**, and cannot modify browser sessions. Local branch metadata writes are the only filesystem modifications it performs.

## How it works

```text
Jan main chat (parent context)
       |
       | branch_solve: small independent problem
       v
Branch MCP server --> Jan local API --> isolated branch model conversation
       |                                |
       |                                +-- optional nested branch
       |                                +-- optional read-only file tools
       v
Writes local compact outcome; optional transcript
       |
       v
Returns branch_id + short outcome to Jan main chat
       |
       v
Jan continues main plan (branch details not imported)
```

## Development tests

Python 3.10+ (tested with current Python 3.x); no third-party dependencies:

```bash
python3 -m unittest discover -s tests -v
```

Tests use a local mock API and verify branch creation, isolation, resuming, nested identifiers, secure file path handling, file permissions, and MCP JSON-RPC shape. The tests do **not** demonstrate a real model choosing to branch automatically; that behavior must be verified in Jan with your selected model.

## Troubleshooting

- **`Cannot reach Jan API`**: enable the Local API Server; test `python3 branch_context.py doctor` from the same machine.
- **`HTTP 401`**: set the API key in `config.json` or `JAN_BRANCH_API_KEY`.
- **`HTTP 404` / no model**: use exact model ID from `/v1/models` and verify model is loaded.
- **MCP server fails immediately**: Jan cannot spawn the Python interpreter/script, or Flatpak permissions prevent access; test `python3 branch_context.py mcp` from a terminal (it deliberately waits silently for MCP input).
- **Tool never called**: enable tool calling and paste assistant instructions; model must follow tool calling.
- **MCP tool timeout**: raise Jan's MCP tool-call timeout; reduce max_agent_turns or max_output_tokens; ensure Jan local model inference is stable.
- **VRAM limit**: this project **reuses** Jan's inference endpoint. Avoid a second independent Jan worker loaded with another model if VRAM is tight. The local API model and Jan desktop model configuration may affect memory usage.
