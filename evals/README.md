# evals/

Benchmark data and suites for the eval harness (M8). The harness **code** lives in the
importable package [`keel.eval`](../src/keel/eval); this directory holds the data it runs on.

## Run it

```bash
# the offline sample suite (no download needed)
keel eval --benchmark local --suite evals/samples --model <mlx-model-id>

# the aider polyglot benchmark (clone it first)
git clone https://github.com/Aider-AI/polyglot-benchmark
keel eval --benchmark aider --suite polyglot-benchmark --model <mlx-model-id> --limit 20

# SWE-bench Verified from a JSONL export (see below) — execution is EXPERIMENTAL
keel eval --benchmark swebench --suite swebench_verified.jsonl --model <mlx-model-id> --limit 5
```

Each run prepares a fresh workspace per case under `runs/`, runs the agent, and scores it by
**execution** (the case's `verify_command`; exit 0 == pass). Protected files (the tests) are
restored before verifying so the agent can't game them. Every case writes a `TraceRecord`, and
`--output report.json` saves the aggregate (pass rate, mean steps, tool-call error rate).

Opus / frontier reference numbers come only from **published leaderboards**, stored as a static
table — never a live API call at runtime (strictly-local invariant). Metrics are reported **at
equal token budgets**.

## Benchmarks

- **`local`** — `LocalBenchmark`: a directory of self-contained cases. Each subdir has a
  `case.json` (`instructions`, `verify_command`, optional `protected_paths`, `setup_commands`);
  every other file is seeded into the workspace. `evals/samples/` ships two tiny Python cases so
  the harness runs offline. Use this format for quick regression cases.
- **`aider`** — `AiderPolyglotBenchmark`: maps the
  [aider polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) Exercism exercises
  into cases. v1 scores **Python** with pytest; other languages are skipped until their runners
  are wired. The cheap, fast regression check.
- **`swebench`** — `SWEBenchVerified`: loads instances from a JSONL export of
  `princeton-nlp/SWE-bench_Verified`. **Execution is experimental**: a case clones the repo at
  `base_commit`, applies the test patch, and runs the FAIL_TO_PASS tests, but Keel does not yet
  provision each repo's environment (the official harness uses per-repo Docker images), so cases
  only run where the toolchain/deps already exist. Today it's for loading and inspecting
  instances; full containerised scoring is not implemented yet.

  Export the dataset to JSONL once:
  ```python
  from datasets import load_dataset

  load_dataset("princeton-nlp/SWE-bench_Verified", split="test").to_json("swebench_verified.jsonl")
  ```

Remember to strip future commits from eval repos (anti "git hacking") when wiring real
SWE-bench execution.
