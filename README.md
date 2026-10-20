<h1 align="center">Keel</h1>

<p align="center">
  <em>A strictly-local, open-source coding-agent framework that makes a small open model
  punch at frontier level — on your laptop, with zero cloud calls.</em>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="Platform: Apple Silicon" src="https://img.shields.io/badge/platform-Apple%20Silicon-black.svg">
</p>

---

## The bet

Most agent harnesses assume a frontier model: long context, reliable tool calls, deep
planning. Drop a small (~3B-active) local model into them and it falls over — context
overflow, broken tool JSON, lost plans. The gap is largely a **harness problem**, not a
parameter problem.

**Keel is the harness, co-designed for a small model's limits.** The keel is what keeps a
small vessel upright: here, the system around the model — not the model's size — is what
keeps it from capsizing.

- 🔒 **Strictly local.** No API calls at inference time. Ever. Escalation means a bigger
  *local* model, more *local* samples, or asking you.
- 📐 **Built for 32K.** Context folding and sub-agents are core, not optional.
- 🧠 **Self-evolving skills.** A ranked skill library that writes new skills from successful
  runs and prunes the ones that fail validation.
- ✅ **Proven with numbers.** Every module ships with an ablation and must beat a plain
  single-agent loop on the *same* model.
- 🍎 **Zero-config on Apple Silicon.** One command profiles your machine and picks a tuned
  local model — no quantization or KV-cache knowledge needed.

## North-star

Percent of Opus-level task success on coding benchmarks (SWE-bench Verified, Terminal-Bench,
Aider polyglot), running **100% on an M4 Max 64GB with zero cloud calls**.

## Requirements

Apple Silicon Mac (M-series), macOS, Python 3.11+. Keel is Apple-Silicon-only — no
CUDA/Linux/Windows support.

## Install

```bash
git clone https://github.com/dhirendrachoudhary/keel
cd keel
pip install -e ".[mlx]"     # the 'mlx' extra pulls the Apple-Silicon inference backend
```

## Usage

```bash
# Profile the machine, pick a model/quant that fits with headroom, and write the config.
keel setup

# Run the coding agent on a task (uses the model from `keel setup`, or pass --model).
keel run "make the failing test in tests/ pass"

# Score a benchmark locally and write a report.
keel eval --benchmark local --suite evals/samples --output report.json
```

`keel setup` inspects your hardware and recommends a local model — for example, on a 64GB
M4 Max it selects a ~30B MoE at int4 (~3B active) that fits in the memory budget with KV
headroom for several concurrent agents. Weights download on first run.

## Architecture

Keel is nine swappable modules behind clean interfaces. The orchestrator plans and hands
**typed subtasks** to workers; every agent shares four services and one local model server.
Only the planner thinks; workers run with thinking off to keep tool calls stable.

| Module | Responsibility |
|---|---|
| Zero-config serving | Profile hardware, pick model/quant, benchmark and tune a local server |
| Agent loop & tools | A tiny auditable loop where a small model never emits a broken tool call |
| Orchestrator | Decompose tasks into typed subtasks; dispatch workers in fresh contexts |
| Context manager | Branch, fold, summarize — never exceed 32K active tokens |
| Skill system | Load, rank, write-from-traces and validation-gate a skill library |
| Harness adaptation | Tune the harness to whichever local model you run |
| Verifier | Sample K candidates and pick the right one (execution + execution-free) |
| Eval & trace store | One trace schema and a reproducible, execution-based eval harness |
| Safety & sandboxing | Workspace-scoped FS, deny-by-default tool network, dry-run, skill scanning |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design in depth.

## Development

```bash
pip install -e ".[dev]"
make check     # ruff + mypy + pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).
