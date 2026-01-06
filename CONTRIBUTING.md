# Contributing to ScaleScope

Thank you for your interest in contributing! This document explains how to set up your environment, coding standards, and Git hooks.

## Getting started

1. Requirements
   - Python 3.11
   - Git

2. Clone and install (editable)
   ```bash
   git clone https://github.com/lightmatter-ai/scalescope.git
   cd scalescope
   python3 -m pip install -e .
   # Optional: install dev tools like pytest
   python3 -m pip install pytest
   ```

3. Run the test suite
   ```bash
   pytest -q
   # Hot Interconnects’25 tests:
   pytest -q tests/test_hot_interconnects.py
   ```

## Code style

Python code style and tooling are configured in `pyproject.toml`:

- Formatter: Black (line-length 100, target Python 3.11)
- Import sorter: isort (profile = black)
- Lint: flake8 (selected ignores, max-line-length 100)

General rules:
- Prefer descriptive names and explicit types for public interfaces.
- Avoid deep nesting; prefer guard clauses.
- Add comments only when non-obvious context or invariants must be communicated.

## Git hooks with pre-commit

This repository includes a `.pre-commit-config.yaml`. To enable hooks locally:

```bash
pip install pre-commit
pre-commit install
# (Optional) run on all files once
pre-commit run --all-files
```

## Commit and PR guidelines

- Keep commits focused and well-titled. Conventional Commits are welcome (e.g., `feat:`, `fix:`, `docs:`).
- Include tests when adding features or fixing bugs.
- Update documentation (README/CONTRIBUTING) when user-facing behavior changes.
- Open PRs against the default branch; include a clear description, motivation, and screenshots/logs if applicable.

## Project structure (high-level)

- `scalescope/`: Python package
  - Core modeling modules: compute, memory, comms, specs, topology
- `model_zoo/`, `train_specs/`, `inference_prefill_specs/`, `inference_decode_specs/`: YAML configs
- `topology_zoo/`: physical and logical topology definitions
- `hardware_specs/`: hardware component and node library
- `tests/`: unit/integration tests

## Troubleshooting

- If pre-commit modifies files unexpectedly: re-run `pre-commit run --all-files` and commit the changes.
- If linters disagree: follow Black’s output, then isort, then flake8 fixes.
- Ensure you are using Python 3.11 when running tools and tests.

## Questions

Open a GitHub discussion or issue with a clear description, reproduction steps, and environment details. We appreciate your contributions!
