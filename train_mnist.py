import argparse
from pathlib import Path

from torch import nn
import torch

from src.models.simple_mlp import SimpleMLP
from src.data.mnist import build_mnist_dataloaders
from src.engine.trainer import train_one_epoch, evaluate
from src.utils.env import get_device
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_checkpoint, save_metrics
from src.utils.logger import SimpleLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleMLP on MNIST")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/mnist_exp001")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    output_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    logger = SimpleLogger(output_dir / "train.log")

    logger.log("========== Experiment Config ==========")
    logger.log(str(vars(args)))
    logger.log(f"Using device: {device}")

    train_loader, test_loader = build_mnist_dataloaders(
        root=args.data_root,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    all_metrics = {
        "config": vars(args),
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        all_metrics["epochs"].append(epoch_metrics)

        logger.log(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4%}"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            ckpt_dir / f"mnist_epoch_{epoch}.pt",
        )

    save_metrics(all_metrics, output_dir / "metrics.json")
    logger.log("Training finished.")


if __name__ == "__main__":
    main()