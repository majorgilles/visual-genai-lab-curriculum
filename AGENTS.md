# AGENTS.md

Guidance for agents and humans working in this repository.

## Intent

This is a learning-first visual generative AI lab. Favor clarity, official tutorial fidelity, and human inspection over
clever abstractions.

## Rules

- Use `uv` for all Python commands.
- Target Windows native PowerShell commands in documentation and issues.
- Keep scripts small and readable.
- Add Python type hints to new or changed code.
- Prefer official sources: uv, PyTorch, torchvision, Hugging Face, and diffusers.
- Do not add classifier-focused curriculum work to the main path.
- Save generated artifacts under `outputs/` and checkpoints under `models/`.
- Every learning issue should remain human-in-the-loop: inspect outputs, tweak one parameter, and write a short note.

## Quality checks

Run before committing code changes:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
