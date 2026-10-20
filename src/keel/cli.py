"""Keel command-line interface.

Surface: ``setup | doctor | run | eval | ablate``. ``setup``, ``run`` and ``eval`` do real
work; ``doctor`` and ``ablate`` are not implemented yet. Nothing here ever makes a cloud
call — Keel is strictly local.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from keel import __version__


def _stub(command: str) -> int:
    print(f"keel {command}: not implemented yet.", file=sys.stderr)
    return 2


def _cmd_setup(write_config: bool = True, agents: int = 3) -> int:
    from keel.config import CONFIG_PATH, KeelConfig, ServingProfile
    from keel.serving.picker import pick_model
    from keel.serving.profiler import detect

    p = detect()
    bandwidth = f"{p.memory_bandwidth_gbps:g} GB/s" if p.memory_bandwidth_gbps else "unknown"
    throttle = "  (THROTTLED)" if p.thermal_throttled else ""
    print("Detected hardware:")
    print(f"  chip:              {p.chip or 'unknown'}")
    print(f"  apple silicon:     {p.is_apple_silicon}")
    print(f"  unified memory:    {p.unified_memory_gb:g} GB")
    print(f"  perf / eff cores:  {p.performance_cores} / {p.efficiency_cores}")
    print(f"  memory bandwidth:  {bandwidth}")
    print(f"  free disk:         {p.free_disk_gb:g} GB")
    print(f"  low power mode:    {p.low_power_mode}")
    print(f"  cpu speed limit:   {p.cpu_speed_limit_pct}%{throttle}")
    print(f"  usable for models: ~{p.usable_model_memory_gb:g} GB")
    print()
    if not p.is_apple_silicon:
        print("warning: Keel targets Apple Silicon; detected a non-Apple chip.", file=sys.stderr)

    pick = pick_model(p, n_agents=agents)
    if pick is None:
        print("No catalogued model fits the usable memory budget.")
        print("Run a smaller model directly with `keel run --model <mlx-id>`, or edit the catalog.")
        return 0

    print("Recommended model:")
    print(f"  {pick.model_id}  [{pick.quantization}, backend {pick.backend}]")
    print(f"  {pick.rationale}")

    if write_config:
        cfg = KeelConfig(
            serving=ServingProfile(
                model_id=pick.model_id, quantization=pick.quantization, backend=pick.backend
            )
        )
        cfg.save()
        print(f"\nWrote {CONFIG_PATH} — `keel run` and `keel eval` will use this model by default.")
        print("(the weights are not downloaded yet; that happens on first run.)", file=sys.stderr)
    else:
        print("\n(--no-write: config not saved.)")

    print("\nNot yet implemented: speed-feature tuning, benchmark/auto-tune, and `keel doctor`.")
    return 0


def _cmd_run(task: str, model: str | None, max_steps: int, workspace: str) -> int:
    from keel.baseline import BaselineAgent
    from keel.config import CONFIG_PATH, KeelConfig
    from keel.serving.backends import create_backend
    from keel.trace import TraceStore

    model_id = model
    if not model_id and CONFIG_PATH.exists():
        model_id = KeelConfig.load().serving.model_id
    if not model_id:
        print(
            "no model configured. Run `keel setup` first, or pass --model <mlx-model-id>.",
            file=sys.stderr,
        )
        return 2

    backend = create_backend("mlx-lm", model_id)
    agent = BaselineAgent(backend, workspace=workspace, max_steps=max_steps)
    result = agent.run(task, model_id=model_id)

    print(f"[{result.outcome} in {result.steps} step(s)]")
    if result.final_message:
        print(result.final_message)
    if result.trace is not None:
        TraceStore().append(result.trace)
        print("(trace appended to traces/keel.trace.jsonl)", file=sys.stderr)
    return 0 if result.outcome == "submitted" else 1


def _cmd_eval(
    benchmark: str,
    suite: str | None,
    model: str | None,
    max_steps: int,
    limit: int | None,
    output: str | None,
    runs_dir: str,
) -> int:
    import json
    from pathlib import Path

    from keel.config import CONFIG_PATH, KeelConfig
    from keel.eval import (
        AiderPolyglotBenchmark,
        Benchmark,
        LocalBenchmark,
        SWEBenchVerified,
        baseline_agent_fn,
        run_benchmark,
    )
    from keel.serving.backends import create_backend
    from keel.trace import TraceStore

    if not suite:
        print(
            "eval needs a path: --suite <case-dir | polyglot-repo | swebench.jsonl>.",
            file=sys.stderr,
        )
        return 2

    bench: Benchmark
    if benchmark == "local":
        bench = LocalBenchmark(suite)
    elif benchmark == "aider":
        bench = AiderPolyglotBenchmark(suite)
    elif benchmark == "swebench":
        bench = SWEBenchVerified(suite)
    else:  # pragma: no cover - argparse restricts choices
        print(f"unknown benchmark {benchmark!r}", file=sys.stderr)
        return 2

    model_id = model
    if not model_id and CONFIG_PATH.exists():
        model_id = KeelConfig.load().serving.model_id
    if not model_id:
        print(
            "no model configured. Run `keel setup` first, or pass --model <mlx-model-id>.",
            file=sys.stderr,
        )
        return 2

    backend = create_backend("mlx-lm", model_id)
    agent_fn = baseline_agent_fn(backend, model_id=model_id, max_steps=max_steps)
    report = run_benchmark(
        bench, agent_fn, runs_dir=runs_dir, model_id=model_id, trace_store=TraceStore(), limit=limit
    )

    print(report.summary_line())
    for r in report.results:
        flag = "PASS" if r.passed else "fail"
        extra = f" — {r.error}" if r.error else ""
        print(f"  {flag}  {r.case_id}  ({r.steps} steps, verify={r.verify_exit_code}){extra}")
    if output:
        Path(output).write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"(report written to {output})", file=sys.stderr)
    return 0 if report.n_total > 0 and report.n_passed == report.n_total else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keel",
        description="Strictly-local small-model coding-agent framework.",
    )
    parser.add_argument("--version", action="version", version=f"keel {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    st = sub.add_parser("setup", help="detect hardware, pick a model/quant, write the config")
    st.add_argument("--no-write", action="store_true", help="don't write .keel/config.json")
    st.add_argument(
        "--agents", type=int, default=3, help="concurrent agents to budget KV for (default 3)"
    )
    sub.add_parser("doctor", help="diagnose slow local inference")
    run = sub.add_parser("run", help="run the baseline coding agent on a task")
    run.add_argument("task", help="the task for the agent to perform")
    run.add_argument("--model", help="mlx-lm model id (else read from .keel/config.json)")
    run.add_argument("--max-steps", type=int, default=30, help="max agent steps (default 30)")
    run.add_argument("--workspace", default=".", help="working directory for the agent")
    ev = sub.add_parser("eval", help="run a benchmark locally and score it")
    ev.add_argument("--benchmark", choices=["local", "aider", "swebench"], default="local")
    ev.add_argument(
        "--suite", help="path: a local case dir, an aider polyglot repo, or a SWE-bench JSONL"
    )
    ev.add_argument("--model", help="mlx-lm model id (else read from .keel/config.json)")
    ev.add_argument(
        "--max-steps", type=int, default=30, help="max agent steps per case (default 30)"
    )
    ev.add_argument("--limit", type=int, help="run at most N cases")
    ev.add_argument("--output", help="write the JSON report to this path")
    ev.add_argument("--runs-dir", default="runs", help="where per-case workspaces are created")
    sub.add_parser("ablate", help="run an ablation that proves a module earns its place")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command: str | None = args.command
    if command is None:
        build_parser().print_help()
        return 0
    if command == "setup":
        return _cmd_setup(write_config=not args.no_write, agents=args.agents)
    if command == "run":
        return _cmd_run(args.task, args.model, args.max_steps, args.workspace)
    if command == "eval":
        return _cmd_eval(
            args.benchmark,
            args.suite,
            args.model,
            args.max_steps,
            args.limit,
            args.output,
            args.runs_dir,
        )
    return _stub(command)


if __name__ == "__main__":
    raise SystemExit(main())
