"""Small uv + PyTorch + CUDA smoke test with a seaborn visual artifact."""

from __future__ import annotations

import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_demo_tensor(device: torch.device) -> torch.Tensor:
    """Create a tiny tensor on the selected device and do simple GPU-friendly math."""
    values = torch.linspace(-3.0, 3.0, steps=64, device=device).reshape(8, 8)
    return torch.sin(values) * torch.cos(values.T)


def save_heatmap(tensor: torch.Tensor, output_path: Path) -> None:
    """Save a heatmap artifact for human inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(tensor.detach().cpu().numpy(), cmap="mako", ax=axis, cbar=True)
    axis.set_title("uv + PyTorch + CUDA smoke-test tensor")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the smoke test and save a visual artifact."""
    device = get_device()
    tensor = build_demo_tensor(device)
    output_path = Path("outputs/prep/cuda_smoke_heatmap.png")
    save_heatmap(tensor, output_path)

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Artifact: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
