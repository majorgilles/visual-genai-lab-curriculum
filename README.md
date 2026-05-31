# Visual GenAI Lab

A small **uv + PyTorch** project for learning visual generative AI through official tutorials and human-in-the-loop experiments.

The curriculum starts with a local Windows-native CUDA smoke test, then moves through official PyTorch generative examples
before following the Hugging Face Diffusion Models Course page by page.

## Goals

- Use `uv` for reproducible Python project management.
- Use CUDA-enabled PyTorch locally on an NVIDIA RTX 4070 Super.
- Keep every learning issue human-in-the-loop: run it, inspect the outputs, tweak one parameter, and write down what changed.
- Prefer official sources first: uv, PyTorch, torchvision, Hugging Face, and diffusers docs/tutorials.
- Produce visible artifacts such as image grids, stylized images, generated samples, and short notes.

## Setup on Windows PowerShell

Install uv if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create/sync the environment:

```powershell
uv sync
```

Run linting and tests:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Prep 1: uv + PyTorch + CUDA smoke test

Run the local CUDA prep script:

```powershell
uv run python -m visual_genai_lab.cuda_smoke
```

Expected result:

- Prints Python, PyTorch, CUDA availability, and GPU name.
- Saves a small seaborn heatmap artifact to `outputs/prep/cuda_smoke_heatmap.png`.
- Uses the GPU for a tiny tensor operation when CUDA is available.

Validated locally on 2026-05-31:

- Python: `3.12.0`
- PyTorch: `2.11.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4070 SUPER`
- CUDA runtime: `12.8`
- Artifact: `outputs/prep/cuda_smoke_heatmap.png`

## Project structure

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── data/                  # local datasets; ignored except .gitkeep
├── models/                # local checkpoints; ignored except .gitkeep
├── notebooks/             # learning notebooks
├── outputs/               # generated artifacts; ignored except selected prep artifacts
├── src/visual_genai_lab/  # small reusable scripts/helpers
└── tests/                 # lightweight tests
```

## Curriculum backbone

### Prep projects

1. **Environment smoke test: uv + PyTorch + CUDA on Windows**
   - Sources: official uv docs and official PyTorch Start Locally docs.
2. **Neural Style Transfer with VGG19**
   - Source: <https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html>
3. **DCGAN image generator**
   - Source: <https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>
4. **VAE latent-space image sampler**
   - Source: <https://github.com/pytorch/examples/tree/main/vae>

### Hugging Face Diffusion Course

After the prep projects, follow the Hugging Face Diffusion Models Course one page at a time:

<https://huggingface.co/learn/diffusion-course/>

Each issue should include a human checkpoint before closing:

- confirm the official source was followed,
- run code locally or document why a notebook/platform was used,
- save a visual artifact,
- tweak one parameter,
- add a short note explaining what changed and what was learned.

## Official references

- uv docs: <https://docs.astral.sh/uv/>
- PyTorch Start Locally: <https://pytorch.org/get-started/locally/>
- PyTorch tutorials: <https://docs.pytorch.org/tutorials/>
- Hugging Face Diffusion Course: <https://huggingface.co/learn/diffusion-course/>
- Hugging Face diffusers docs: <https://huggingface.co/docs/diffusers>
