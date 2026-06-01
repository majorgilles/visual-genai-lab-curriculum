# Visual GenAI Lab

A small **applied machine learning lab** for learning visual generative AI with **uv + PyTorch**, official tutorials, runnable local code, and human-in-the-loop inspection.

This is an applied ML / visual AI engineering curriculum, not a pure theory track. The focus is the practical loop used in real AI work: prepare inputs, run a model, inspect outputs, tweak one thing, save an artifact, and write down what changed. The curriculum starts with a local Windows-native CUDA smoke test, adds VGG19 image-recognition bridge labs, then moves through official PyTorch generative examples before following the Hugging Face Diffusion Models Course page by page.

## Goals

- Build practical fluency with the applied ML workflow: input data → model → output artifact → human review note.
- Use `uv` for reproducible Python project management.
- Use CUDA-enabled PyTorch locally on an NVIDIA RTX 4070 Super.
- Learn pretrained-model reuse before deeper generative work: official preprocessing, inference, labels, feature extractors, and visible outputs.
- Keep every learning issue human-in-the-loop: run it, inspect the outputs, tweak one parameter, and write down what changed.
- Prefer official sources first: uv, PyTorch, torchvision, Hugging Face, and diffusers docs/tutorials.
- Produce visible artifacts such as prediction grids, image grids, stylized images, generated samples, and short notes.

## Learning, authorship, and AI assistance

This project is AI-assisted as a learning tool. AI support is used to explore questions and ideas, organize the learning path, and suggest code and notes while studying CNN filters, feature maps, pretrained vision models, and visual generative AI workflows.

Code in this repository is hand-written as part of the learning process, based on AI suggestions, with the exception of some `matplotlib.pyplot` graphing code used to generate visualizations. Written explanations in this repo may be AI-summarized from my prompts, inspection, and learning process; they are not presented as my unaided wording.

The inspiration, direction, and curiosity behind the project are mine. This repo is a vehicle for gaining knowledge about visual ML through inspection, experimentation, and reflection.

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
- Baseline artifact: `outputs/prep/cuda_smoke_heatmap.png`
- Parameter-tweak artifact: `outputs/prep/cuda_smoke_heatmap_span6.png`

Human review note for issue #1:

- I inspected both heatmaps and confirmed they render as complete 8×8 seaborn PNG artifacts with color bars and clear titles.
- I reran the smoke test with a parameter tweak, increasing the demo tensor span from `3` to `6`:

  ```powershell
  uv run python -m visual_genai_lab.cuda_smoke --span 6 --output outputs/prep/cuda_smoke_heatmap_span6.png
  ```

- Observation: increasing `--span` changes the sinusoidal pattern substantially while preserving the same tensor shape and CUDA execution path, which confirms the artifact is generated from live PyTorch tensor math rather than a static image.

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
2. **Pretrained VGG19 image-recognition inference**
   - Source: <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html>
   - GitHub issue: <https://github.com/majorgilles/visual-genai-lab-curriculum/issues/17>
3. **Frozen-feature VGG19 image classifier**
   - Sources: <https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html> and <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html>
   - GitHub issue: <https://github.com/majorgilles/visual-genai-lab-curriculum/issues/18>
4. **Neural Style Transfer with VGG19**
   - Source: <https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html>
   - GitHub issue: <https://github.com/majorgilles/visual-genai-lab-curriculum/issues/2>
5. **DCGAN image generator**
   - Source: <https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>
   - GitHub issue: <https://github.com/majorgilles/visual-genai-lab-curriculum/issues/3>
6. **VAE latent-space image sampler**
   - Source: <https://github.com/pytorch/examples/tree/main/vae>
   - GitHub issue: <https://github.com/majorgilles/visual-genai-lab-curriculum/issues/4>

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
