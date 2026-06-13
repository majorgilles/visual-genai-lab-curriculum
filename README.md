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

## Prep 4: Neural Style Transfer with VGG19

Source: <https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html>  
GitHub issue: <https://github.com/majorgilles/visual-genai-lab-curriculum/issues/2>

Runnable notebook:

```powershell
uv run jupyter lab notebooks/03_neural_style.ipynb
```

Artifacts saved:

- `outputs/prep/style_transfer/content_dancing.png`
- `outputs/prep/style_transfer/style_picasso.png`
- `outputs/prep/style_transfer/stylized_noise_init.png`
- `outputs/prep/style_transfer/style_transfer_noise_gallery.png`

Human review note for issue #2:

- I followed the official PyTorch Neural Transfer tutorial closely in `notebooks/03_neural_style.ipynb`.
- I inspected the content image, style image, generated output, and saved gallery artifact.
- I used pure noise as the starting `input_img` instead of the tutorial's `content_img.clone()` initialization:

  ```python
  input_img = torch.randn(content_img.data.size(), device=device)
  ```

- I also ran a longer optimization pass with `num_steps=600`.
- Observation: even from random noise, the optimizer recovered recognizable dancer/content structure while adding Picasso-like color and texture patterns. This made the feature-space loss idea concrete: VGG19 acted as a fixed visual judge, while the generated image pixels changed.
- Decision: the result is good enough to count as a successful neural style transfer artifact for issue #2.

## Hugging Face Diffusion Course Unit 1: Introduction to Diffusers

Source: <https://huggingface.co/learn/diffusion-course/unit1/2>

GitHub issue: <https://github.com/majorgilles/visual-genai-lab-curriculum/issues/7>

Runnable notebook:

```powershell
uv run jupyter lab notebooks/06_hf_diffusion_unit1_intro.ipynb
```

Artifacts saved:

- `outputs/hf-course/unit1_generated_samples_seed0_steps100.png`
- `outputs/hf-course/unit1_mvp_butterflies_seed7_steps100.png`
- `outputs/hf-course/unit1_mvp_butterflies_seed13_steps100.png`
- `outputs/hf-course/unit1_pretrained_butterflies_grid.png`

Human review note for issue #7:

- I followed the official Hugging Face Diffusion Course Unit 1 introduction in `notebooks/06_hf_diffusion_unit1_intro.ipynb` while adapting it to this local `uv` project.
- I loaded the Smithsonian butterflies dataset, preprocessed images into `[batch, 3, 32, 32]` tensors, added noise with `DDPMScheduler`, trained a small `UNet2DModel` to predict noise, and generated samples with `DDPMPipeline`.
- I inspected the generated sample grid at `outputs/hf-course/unit1_generated_samples_seed0_steps100.png`.
- Observation: the images display correctly and are recognizable, noisy, butterfly-like images.
- Parameter tweak: seed-based sample variants are saved under `outputs/hf-course/`; changing the seed changes the generated butterflies while keeping the same `100` inference-step sampling setup.
- Decision: good enough to continue; the local diffusion workflow runs end to end, even though the tiny training run still produces rough images.

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
