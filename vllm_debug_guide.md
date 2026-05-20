# vLLM Request Lifecycle — Hands-on Debugging Guide

Companion to `/root/request(2).md`. Walks you through running a real request
through vLLM and stepping into the stages described in that doc using Cursor's
debugger.

---

## What's already set up

- vLLM source: `/root/vllm` (editable installed, on `main`)
- Python venv: `/root/vllm/.venv` (Python 3.12)
- Model: `/root/models/Qwen3-0.6B` (~1.5GB)
- Torch 2.11.0 + CUDA 12.8 (Blackwell-compatible for RTX 5090)

Sanity check before starting:

```bash
source /root/vllm/.venv/bin/activate
python -c "import vllm; print(vllm.__file__)"
# Must print: /root/vllm/vllm/__init__.py
# If it points into site-packages, breakpoints won't fire in your source tree.
```

---

## Step 1 — Open vLLM in Cursor

Open `/root/vllm` as a **new Cursor workspace** (separate window from
`/root/pytorch`). Mixing the two leads to wrong Python interpreter selection
and confusing import errors.

In Cursor, select the interpreter: `Cmd/Ctrl+Shift+P` → "Python: Select
Interpreter" → `/root/vllm/.venv/bin/python`.

---

## Step 2 — Create `.vscode/launch.json`

Path: `/root/vllm/.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "vLLM: offline trace (Qwen3-0.6B)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/trace_request.py",
      "console": "integratedTerminal",
      "justMyCode": false,
      "python": "${workspaceFolder}/.venv/bin/python",
      "env": {
        "VLLM_USE_V1": "1",
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "CUDA_VISIBLE_DEVICES": "0"
      }
    }
  ]
}
```

**Critical**: `"justMyCode": false`. Without it, the debugger silently skips
over vLLM internals (treats them as library code) and **none of your
breakpoints will fire**. This is the #1 gotcha.

---

## Step 3 — Create the trace script

Path: `/root/vllm/trace_request.py`

```python
"""Minimal offline request to step through the vLLM request lifecycle."""
from vllm import LLM, SamplingParams

llm = LLM(
    model="/root/models/Qwen3-0.6B",
    dtype="bfloat16",
    gpu_memory_utilization=0.5,   # leave headroom on the 5090
    max_model_len=2048,
    enforce_eager=True,           # disable CUDA graphs — required for steppable forward
)

prompts = ["Explain what a transformer is in one sentence."]
sampling = SamplingParams(temperature=0.7, max_tokens=32)

outputs = llm.generate(prompts, sampling)
for out in outputs:
    print("PROMPT :", out.prompt)
    print("OUTPUT :", out.outputs[0].text)
```

Why these flags matter:

| Flag | Why |
|---|---|
| `enforce_eager=True` | Without this, the model forward is replayed as an opaque CUDA graph and your breakpoints inside transformer layers never fire. |
| `gpu_memory_utilization=0.5` | vLLM grabs 90% of VRAM for KV cache by default. For learning, you don't need it and it makes other tools (nvidia-smi) noisier. |
| `max_model_len=2048` | Smaller KV cache = faster startup. |

**Note**: This is the **synchronous offline path** (`LLM` class), not the HTTP
server path. The doc's Stages 1-4 (FastAPI, ZMQ, AsyncLLM) are **skipped**
here — for first contact, fewer processes = simpler debugging. Stages 5-10
are the same. See Step 6 below to graduate to the HTTP path.

---

## Step 4 — Set breakpoints in reading order

Set these in Cursor by clicking the gutter next to the line number. The
**doc's stages map directly** to these functions. Start with just the bolded
two for your first run.

| Doc Stage | File | Function | What to inspect |
|---|---|---|---|
| Stage 4 (submit) | `vllm/v1/engine/llm_engine.py` | `add_request()` | `EngineCoreRequest` fields |
| **Stage 5 (schedule)** | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()` | `running` / `waiting` lists, `num_scheduled_tokens` |
| Stage 6 (input prep) | `vllm/v1/worker/gpu_model_runner.py` | `_prepare_inputs()` | `input_ids`, `positions`, `query_start_loc` GPU tensors |
| **Stage 7 (forward)** | `vllm/v1/worker/gpu_model_runner.py` | `execute_model()` (right before the `model(...)` call) | `inputs_embeds.shape`, hidden states |
| Stage 8 (sample) | `vllm/v1/worker/gpu_model_runner.py` | `sample_tokens()` | `logits` tensor, then `sampled_token_ids` after `_sample()` |
| Stage 9 (state update) | `vllm/v1/core/sched/scheduler.py` | `update_from_output()` | Token appended, `check_stop()` decision |
| Stage 10 (detokenize) | `vllm/v1/engine/output_processor.py` | `process_outputs()` | `EngineCoreOutput` → `RequestOutput` |

**Tip**: Line numbers in `/root/request(2).md` are for vLLM `v0.20.0`; your
checkout is on `main`, so they'll have drifted. **Search by function name**,
not line number. Function names are stable across releases.

---

## Step 5 — Run it

Hit `F5` in Cursor (or click the green "play" icon in the Run/Debug panel).

What you'll see in order:

1. **~30s cold start** — vLLM loads weights, profiles GPU memory, captures
   piecewise CUDA graphs for the attention kernel, warms up.
   Breakpoints **will NOT hit** during this phase. Be patient.
2. Once `llm.generate(...)` is reached, your `Scheduler.schedule()`
   breakpoint fires for the **first iteration (prefill)**.
3. F5 to continue → `execute_model()` fires → forward pass runs.
4. F5 → back to `Scheduler.schedule()` for the **second iteration
   (first decode token)**.
5. This loop continues until 32 tokens generated or EOS.

### Expected behavior, not bugs

- Piecewise CUDA graph warnings during warmup — these are for the attention
  op only; the rest of the model stays eager.
- `Scheduler.schedule()` fires once for prefill, then once per decode token.
  Your breakpoint will hit ~33 times for this script (1 prefill + 32 decode).
- Many local variables show as `<unavailable>` until you step over the line
  that creates them — normal Python debugger behavior.

---

## Step 6 — Mapping what you see to the doc

When you hit `Scheduler.schedule()` (Stage 5), in the Variables panel:

- `self.waiting` — `deque` of new requests waiting for prefill
- `self.running` — list of requests currently generating
- The doc's "Phase 1: schedule RUNNING requests" is the loop starting around
  line 387 in the doc; the equivalent in your code is the `while
  req_index < len(self.running)` loop.
- The doc's "Phase 2: schedule WAITING requests" is the loop that pops from
  `self.waiting`.

When you hit `execute_model()` (Stage 7), the call stack is:

```
trace_request.py:18 (llm.generate)
  → llm_engine.py:add_request
  → llm_engine.py:step
  → executor.execute_model
  → gpu_model_runner.py:execute_model  ← you are here
```

In the offline path, this all runs in **one process**. The doc's Stages 5-9
are spread across EngineCore and Worker processes in the HTTP path, but
collapsed into one call stack here. That's fine — the **logic is identical**,
the **boundary is different**.

---

## Step 7 — Graduate to the HTTP server path (later)

Once you understand the offline flow, run the actual OpenAI-compatible server
to see Stages 1-4 (FastAPI, ZMQ, AsyncLLM):

```bash
source /root/vllm/.venv/bin/activate
vllm serve /root/models/Qwen3-0.6B --enforce-eager --gpu-memory-utilization 0.5
```

In another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/models/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```

### HTTP-path breakpoints (Stages 1-4)

These four stages cover the request *before* it reaches the scheduler. Stages
5-10 in the Step 4 table run the same regardless of offline vs. HTTP — they
just live in different processes under `vllm serve`.

| Doc Stage | File | Function | What to inspect |
|---|---|---|---|
| **Stage 1 (FastAPI route)** | `vllm/entrypoints/openai/chat_completion/api_router.py` | `create_chat_completion()` | `raw_request` headers, `ChatCompletionRequest` body |
| Stage 2 (OpenAI handler) | `vllm/entrypoints/openai/chat_completion/serving.py` | `_create_chat_completion()` | chat templating, sampling params, `request_id` |
| **Stage 3 (AsyncLLM)** | `vllm/v1/engine/async_llm.py` | `AsyncLLM.add_request()` | input processing, output queue setup |
| Stage 4 (ZMQ submit) | `vllm/v1/engine/core_client.py` | `AsyncMPClient.add_request_async()` | `EngineCoreRequest` sent across processes to EngineCore |

(Use `DPAsyncMPClient.add_request_async()` in the same file instead of
`AsyncMPClient` if you serve with `--data-parallel-size > 1`.)

Debugging the HTTP path is harder because there are **3 processes** (API,
EngineCore, Worker) and ZMQ between them. Two options:

**Option A — force single process** (easiest, and required for Stages 5-10
breakpoints to be usable under `vllm serve`):

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve /root/models/Qwen3-0.6B \
  --enforce-eager --gpu-memory-utilization 0.5
```

Everything runs in one process, breakpoints work normally. Without this env
var, breakpoints inside EngineCore (Stages 5, 9, 10) and the Worker (Stages
6, 7, 8) drop into pdb in a subprocess that has no controlling TTY — the
subprocess hangs silently. Trade-off: not representative of production
architecture. Stages 1-4 always run in the API server (foreground) process,
so they're fine either way.

**Option B — attach to a specific process**:

Add near the top of `vllm/v1/engine/core.py`'s `EngineCoreProc.__init__`:

```python
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

Then use Cursor's "Python: Attach" configuration on port 5678. You can only
attach to one process at a time this way.

---

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| Breakpoints don't fire | `justMyCode: true` in launch.json | Set to `false` |
| `vllm.__file__` points to site-packages | Editable install failed | Re-run `VLLM_USE_PRECOMPILED=1 uv pip install -e .` |
| Breakpoint in `model.forward()` doesn't fire | CUDA graphs replayed | Add `enforce_eager=True` to `LLM(...)` |
| "Out of memory" on startup | Default 90% VRAM allocation | Set `gpu_memory_utilization=0.5` |
| First breakpoint never hits | Warmup phase still running | Wait ~30s for cold start to complete |
| Step Into (F11) skips into C code | Stepped into a Triton/CUDA kernel | Step Out (Shift+F11), use Step Over (F10) instead |

---

## Suggested first session (~30 minutes)

1. Open `/root/vllm` in Cursor, set the interpreter.
2. Create `.vscode/launch.json` and `trace_request.py` (Steps 2-3).
3. Set just **two** breakpoints:
   - `Scheduler.schedule()` in `scheduler.py`
   - `execute_model()` in `gpu_model_runner.py`
4. Hit F5. Wait for warmup. When the first breakpoint fires, hover over
   `self.waiting` and `self.running` in the Variables panel.
5. F5 to continue to `execute_model()`. Inspect `scheduler_output`.
6. F5 several more times — watch the same two breakpoints fire repeatedly as
   the decode loop runs.
7. Open `/root/request(2).md` Section 6 ("The Iteration Loop") side by side
   and trace each F5 to a phase in the diagram.

Once that clicks, add breakpoints in `_prepare_inputs()`, `sample_tokens()`,
and `update_from_output()` to see Stages 6, 8, 9 in detail.
