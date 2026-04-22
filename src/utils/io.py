import json
from pathlib import Path

import torch


def ensure_dir(path: str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_checkpoint(model, optimizer, epoch: int, save_path: str) -> None:
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path_obj,
    )


def save_metrics(metrics: dict, save_path: str) -> None:
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path_obj, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)