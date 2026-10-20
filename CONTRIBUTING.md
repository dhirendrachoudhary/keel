# Contributing

Thanks for your interest in Keel. This guide covers local development.

## Setup

Keel targets Apple Silicon (M-series) running macOS with Python 3.11+.

```bash
git clone https://github.com/dhirendrachoudhary/keel
cd keel
pip install -e ".[dev]"      # dev tools: ruff, mypy, pytest
pip install -e ".[mlx]"      # add the inference backend when working on serving/run
```

## Checks

All of these must pass before a change is merged:

```bash
make check     # runs all three:
make lint      # ruff check
make fmt       # ruff format
make typecheck # mypy src/keel
make test      # pytest
```

CI runs the same checks on macOS. The test suite does not download models or make network
calls — inference backends are mocked — so it is fast and hermetic.

## Design principles

- **Strictly local.** Never add a cloud or API dependency on the inference path. If you think
  you need one, you don't — open an issue instead.
- **Interfaces first.** Keep every module behind a swappable interface. Reuse existing engines
  and formats; implement the research ourselves.
- **Prove it.** A module earns its place by beating the plain baseline loop on the same model
  at an equal token budget. Add an ablation with your change.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the overall design.

## Adding things

- **A benchmark adapter** implements `keel.eval.Benchmark` and yields `EvalCase`s. See
  `keel/eval/local.py` for the simplest example.
- **An inference backend** implements `keel.serving.backends.InferenceBackend` and is
  registered in `keel/serving/backends/__init__.py`.
- **A model** is a `ModelSpec` entry in `keel/serving/catalog.py`.

## Style

Code is formatted and linted with ruff and type-checked with mypy (annotations required in
`src/keel`). Match the surrounding code; keep comments at the density of the file you're in.
