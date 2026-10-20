# Architecture

Keel is an open-source, strictly-local coding-agent framework. Its thesis: a small open
model (~3B active) running entirely on an Apple-Silicon laptop can recover most of
frontier-level coding performance if the harness around it is co-designed for its limits.
The win is in system design, not parameters.

## Invariants

These are project-defining. Nothing in the codebase should violate them.

1. **Strictly local at inference time.** No network calls to any model API or remote
   inference endpoint during agent runs. Escalation means a bigger *local* model, more
   *local* samples, or asking the user — never the cloud. Reference numbers for frontier
   models come only from published leaderboards, never from live API calls.
2. **32K active-token cap per agent.** Context folding and sub-agents are core, not optional.
   A budget meter warns at 70%, folds at 85%, and hard-stops at 32K.
3. **Everything is proven with numbers.** Every module ships with an ablation and must beat
   a plain single-agent loop on the same local model at an equal token budget. Every claim
   maps to a reproducible local run.
4. **Reuse engines and formats; implement the research ourselves.** Wrap existing backends
   behind one interface. The value is hardware-aware selection, tuning and diagnosis, and the
   co-designed harness — not a new inference engine.
5. **Fit in memory.** All resident models together target the usable memory budget of the
   host (roughly 75% of unified memory). Prefer MoE (few active params) for speed, and prefer
   one shared model with role prompts before running multiple models.
6. **Safe by default.** A shell-capable, self-writing-skill agent runs with workspace-scoped
   file access, a deny-by-default tool network, dry-run for destructive commands, and skill
   scanning before admission.

## Hardware target

Apple Silicon (M-series), macOS. The reference target is an M4 Max with 64GB unified memory.
No CUDA/Linux/Windows support.

## Modules

The orchestrator plans and hands typed subtasks to workers; every agent shares four services
and one local model server. Only the planner thinks; workers run with thinking off to keep
tool calls stable. Each module sits behind a swappable interface.

### Zero-config serving
`keel setup` profiles the machine (chip, unified memory, bandwidth, cores, free disk, power
and thermal state), picks a model and quantization that fit — weights + KV cache per agent +
headroom — preferring MoE for speed, and writes a single config the rest of the system reads.
`keel doctor` diagnoses slow local inference (a dense model where an MoE would fit, an old
Metal path instead of MLX, memory pressure and swap, Low Power Mode or thermal throttling,
prefill-bound prompts re-sent with no prefix cache, runaway reasoning tokens). The serving
layer wraps existing MLX backends behind one `InferenceBackend` interface.

### Agent loop & tools
A tiny, auditable loop where a small model never emits a broken tool call: constrained
decoding (structural tags) plus argument validation and retries, a tool router that exposes
only a few tools per step, and an evidence-first result contract for every tool — findings,
evidence, unknowns — because small models reason better over explicit evidence. An MCP client
keeps tools pluggable.

### Orchestrator
A planner decomposes a coding task into typed subtasks and dispatches workers in fresh
contexts. The handoff is typed: task, inputs, allowed tools, success check, return schema.
Only the planner thinks; unrestricted thinking destabilizes small-agent tool orchestration.

### Context manager
No agent ever exceeds 32K active tokens, and nothing important is lost. Training-free
rendering dedupes repeats and folds completed turns over an append-only history; a
branch/return mechanism runs a sub-trajectory and returns a short summary; huge inputs (whole
repos, logs) are treated as data queried by code rather than tokens read into context.

### Skill system
The right one to three skills load into a small context at the right time, and the library
improves itself from failures without rotting. Four sub-parts: format and loading (portable
`SKILL.md` with progressive disclosure), ranking and composition, writing new skills from
successful traces, and maintenance via validation-gated pruning and skill-graph health checks.
Skill bodies are capped to a token budget, since every loaded skill eats the 32K window.

### Harness adaptation
The harness tunes itself to whichever local model the user runs, instead of hand-tuning
prompts per model: a meta-search over harness variants (prompts, tool descriptions, step
structure) runs offline on a dev task set when a new model is installed, and the winning
profile is stored next to the serving profile.

### Verifier & test-time scaling
The main lever for closing the gap without a cloud model: sample K candidate solutions and
pick the right one. Verifiers are hybrid — execution-based (tests, lint, type checks) and
execution-free (comparing candidates against each other) — and K adapts to task difficulty
and to the machine's battery and thermal state.

### Eval & trace store
One trace schema (prompt, context folds, tool calls, verifier scores, outcome, metrics) is
shared by skill writing, harness adaptation and later planner training. The eval harness runs
a benchmark in a fresh per-case workspace, scores it by execution, restores protected test
files so the agent cannot game them, and records a trace per case. An ablation runner toggles
each module at equal budgets and fixed seeds.

### Safety & sandboxing
A local agent with shell access and self-written skills must not damage the machine or be
hijacked by a skill: skill scanning before admission, workspace-scoped file access, a
deny-by-default tool network, dry-run for destructive commands, command approval with learned
safe-command memory, and config backups before writing.

## Evaluation

Keel is measured against a plain single-agent loop on the same local model, at equal token
budgets, on standard coding benchmarks: SWE-bench Verified, Terminal-Bench, Aider polyglot,
and a skills benchmark. Reported metrics include pass rate, percent of a frontier model's
score, wall-clock time, tokens, tool-call error rate, escalations per task, and peak memory.
Every module must earn its place.
