from pathlib import Path

import torch

from visual_genai_lab.cuda_smoke import build_demo_tensor, save_heatmap


def test_build_demo_tensor_shape() -> None:
    tensor = build_demo_tensor(torch.device("cpu"))
    assert tensor.shape == (8, 8)


def test_save_heatmap(tmp_path: Path) -> None:
    tensor = build_demo_tensor(torch.device("cpu"))
    output_path = tmp_path / "heatmap.png"
    save_heatmap(tensor, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
