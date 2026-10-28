#!/usr/bin/env python3
"""Prepare extracted wideband IQ and fine-tune a copy of the released XCiT.

The released checkpoint is never overwritten.  Dataset splits are generated
from separate seeded wideband streams, and every sample uses a GT boundary so
the classifier is not trained on incorrectly matched detector predictions.
Use ``benchmark_combined.py --xcit-weights ...`` for the final YOLO-box test.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset
from torchsig_models.models import XCiTClassifier

from benchmark_combined import make_stream
from combined_inference import CLASS_NAMES, extract_iq, signal_family
from inference import as_signal_list, xywh_to_xyxy


ROOT = Path(__file__).resolve().parent
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="generate a GT-box extracted dataset")
    prepare.add_argument("--mixtures", type=int, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--config", type=Path,
                         default=ROOT / "configs/wideband_clean_val_all.yaml")
    prepare.add_argument("--out", type=Path, required=True)

    train = subparsers.add_parser("train", help="fine-tune a checkpoint copy")
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--val-data", type=Path, required=True)
    train.add_argument("--base-checkpoint", type=Path,
                       default=ROOT / "models/xcit_narrowband_v1.0.0.ckpt")
    train.add_argument("--out-checkpoint", type=Path, required=True)
    train.add_argument("--history", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument(
        "--family-loss-weight", type=float, default=0.5,
        help="Weight of the modulation-family loss added to the 57-class loss",
    )
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--seed", type=int, default=20261023)
    return parser


def prepare_dataset(args: argparse.Namespace) -> None:
    if args.mixtures < 1:
        raise ValueError("--mixtures must be positive")
    destination = args.out.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream, capture, metadata, _ = make_stream(args.config, args.seed)
    sample_rate = float(metadata["sample_rate"])
    inputs: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    failures: list[dict] = []
    counts: Counter[str] = Counter()
    started = time.perf_counter()

    for mixture_index in range(args.mixtures):
        capture.clear()
        _, targets = next(stream)[0]
        raw = capture.get("raw")
        if raw is None:
            raise RuntimeError(f"IQ capture missing for mixture {mixture_index}")
        boxes = as_signal_list(targets[0], "box")
        names = [str(name) for name in as_signal_list(targets[1], "name")]
        if len(boxes) != len(names):
            raise RuntimeError(f"box/name mismatch for mixture {mixture_index}")
        for target_index, (label, name) in enumerate(zip(boxes, names)):
            if name not in CLASS_TO_INDEX:
                raise RuntimeError(f"Unknown checkpoint class: {name}")
            _, cx, cy, width, height = map(float, label)
            box = xywh_to_xyxy(cx, cy, width, height)
            try:
                channels, _ = extract_iq(raw, box, sample_rate)
            except (ValueError, RuntimeError) as error:
                failures.append({
                    "mixture_index": mixture_index,
                    "target_index": target_index,
                    "class_name": name,
                    "error": str(error),
                })
                continue
            inputs.append(channels.astype(np.float16))
            labels.append(CLASS_TO_INDEX[name])
            groups.append(mixture_index)
            counts[name] += 1
        if (mixture_index + 1) % 100 == 0 or mixture_index + 1 == args.mixtures:
            print(
                f"[{mixture_index+1}/{args.mixtures}] samples={len(inputs)} "
                f"failures={len(failures)}",
                flush=True,
            )

    if not inputs:
        raise RuntimeError("No extracted training samples were produced")
    np.savez_compressed(
        destination,
        x=np.stack(inputs, axis=0),
        y=np.asarray(labels, dtype=np.int64),
        group=np.asarray(groups, dtype=np.int32),
        class_names=np.asarray(CLASS_NAMES),
        seed=np.asarray(args.seed, dtype=np.int64),
        mixtures=np.asarray(args.mixtures, dtype=np.int64),
    )
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({
            "seed": args.seed,
            "mixtures": args.mixtures,
            "samples": len(inputs),
            "failures": failures,
            "class_counts": dict(sorted(counts.items())),
            "missing_classes": [name for name in CLASS_NAMES if counts[name] == 0],
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "GT boxes from a dedicated wideband stream; production extract_iq",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(inputs)} samples to {destination}", flush=True)


def load_dataset(path: Path) -> tuple[TensorDataset, dict]:
    with np.load(path, allow_pickle=False) as data:
        names = [str(name) for name in data["class_names"].tolist()]
        if names != list(CLASS_NAMES):
            raise RuntimeError(f"Class order mismatch in {path}")
        x = np.asarray(data["x"], dtype=np.float16)
        y = np.asarray(data["y"], dtype=np.int64)
        info = {
            "path": str(path.resolve()),
            "samples": len(y),
            "seed": int(data["seed"]),
            "mixtures": int(data["mixtures"]),
        }
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), info


def family_projection(device: str) -> tuple[list[str], torch.Tensor]:
    families = sorted({signal_family(name) for name in CLASS_NAMES})
    projection = torch.zeros((len(CLASS_NAMES), len(families)), device=device)
    for class_index, name in enumerate(CLASS_NAMES):
        projection[class_index, families.index(signal_family(name))] = 1.0
    return families, projection


def hierarchical_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    projection: torch.Tensor,
    family_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, 57-class, and family losses from one set of logits."""
    fine_loss = functional.cross_entropy(logits, labels)
    family_mask = projection.bool().unsqueeze(0)
    family_logits = torch.logsumexp(
        logits.unsqueeze(2).masked_fill(~family_mask, float("-inf")), dim=1
    )
    family_labels = projection[labels].argmax(dim=1)
    family_loss = functional.cross_entropy(family_logits, family_labels)
    total_loss = fine_loss + family_loss_weight * family_loss
    return total_loss, fine_loss, family_loss


def evaluate(
    model: XCiTClassifier,
    loader: DataLoader,
    device: str,
    projection: torch.Tensor,
    family_loss_weight: float,
) -> dict:
    model.eval()
    total = fine_correct = family_correct = 0
    loss_sum = fine_loss_sum = family_loss_sum = 0.0
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                logits = model(inputs)
                loss, fine_loss, family_loss = hierarchical_losses(
                    logits, labels, projection, family_loss_weight
                )
            probabilities = logits.float().softmax(dim=1)
            predicted = probabilities.argmax(dim=1)
            predicted_family = (probabilities @ projection).argmax(dim=1)
            true_family = projection[labels].argmax(dim=1)
            batch_size = len(labels)
            total += batch_size
            loss_sum += float(loss) * batch_size
            fine_loss_sum += float(fine_loss) * batch_size
            family_loss_sum += float(family_loss) * batch_size
            fine_correct += int((predicted == labels).sum())
            family_correct += int((predicted_family == true_family).sum())
    return {
        "samples": total,
        "loss": loss_sum / total,
        "fine_loss": fine_loss_sum / total,
        "family_loss": family_loss_sum / total,
        "fine_correct": fine_correct,
        "fine_accuracy": fine_correct / total,
        "family_correct": family_correct,
        "family_accuracy": family_correct / total,
    }


def save_checkpoint_copy(
    base_path: Path,
    destination: Path,
    model: XCiTClassifier,
    metadata: dict,
) -> None:
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    checkpoint["joint_finetune"] = metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def train_model(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    if args.family_loss_weight < 0:
        raise ValueError("--family-loss-weight must be non-negative")
    if args.out_checkpoint.resolve() == args.base_checkpoint.resolve():
        raise ValueError("The released base checkpoint must not be overwritten")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0" if args.device == "cuda" else "cpu"
    train_set, train_info = load_dataset(args.train_data)
    val_set, val_info = load_dataset(args.val_data)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=0, pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=args.device == "cuda",
    )
    model = XCiTClassifier.load_from_checkpoint(
        str(args.base_checkpoint), map_location="cpu", weights_only=False
    ).to(device)
    if int(model.hparams.num_classes) != len(CLASS_NAMES):
        raise RuntimeError("Checkpoint class count mismatch")
    _, projection = family_projection(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = {
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "train": train_info,
        "validation": val_info,
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "family_loss_weight": args.family_loss_weight,
            "seed": args.seed,
        },
        "epochs": [],
    }
    started = time.perf_counter()
    baseline = evaluate(
        model, val_loader, device, projection, args.family_loss_weight
    )
    history["baseline_validation"] = baseline
    print("Baseline validation:", baseline, flush=True)
    best_fine = baseline["fine_accuracy"]
    save_checkpoint_copy(
        args.base_checkpoint,
        args.out_checkpoint,
        model,
        {"best_epoch": 0, "validation": baseline,
         "train_data": train_info, "validation_data": val_info,
         "settings": history["settings"]},
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = family_correct = 0
        loss_sum = fine_loss_sum = family_loss_sum = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                logits = model(inputs)
                loss, fine_loss, family_loss = hierarchical_losses(
                    logits, labels, projection, args.family_loss_weight
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = len(labels)
            total += batch_size
            loss_sum += float(loss.detach()) * batch_size
            fine_loss_sum += float(fine_loss.detach()) * batch_size
            family_loss_sum += float(family_loss.detach()) * batch_size
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            probabilities = logits.detach().float().softmax(dim=1)
            predicted_family = (probabilities @ projection).argmax(dim=1)
            true_family = projection[labels].argmax(dim=1)
            family_correct += int((predicted_family == true_family).sum())
        validation = evaluate(
            model, val_loader, device, projection, args.family_loss_weight
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_fine_loss": fine_loss_sum / total,
            "train_family_loss": family_loss_sum / total,
            "train_fine_accuracy": correct / total,
            "train_family_accuracy": family_correct / total,
            "validation": validation,
        }
        history["epochs"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation["fine_accuracy"] > best_fine:
            best_fine = validation["fine_accuracy"]
            save_checkpoint_copy(
                args.base_checkpoint,
                args.out_checkpoint,
                model,
                {"best_epoch": epoch, "validation": validation,
                 "train_data": train_info, "validation_data": val_info,
                 "settings": history["settings"]},
            )
            print(f"Saved new best checkpoint: {args.out_checkpoint}", flush=True)

    history["elapsed_seconds"] = time.perf_counter() - started
    history["best_validation_fine_accuracy"] = best_fine
    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Verify that the produced artifact is loadable by the unchanged inference path.
    verified = XCiTClassifier.load_from_checkpoint(
        str(args.out_checkpoint), map_location="cpu", weights_only=False
    )
    if int(verified.hparams.num_classes) != len(CLASS_NAMES):
        raise RuntimeError("Saved fine-tuned checkpoint failed compatibility check")
    print(f"Training complete; history: {args.history}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_dataset(args)
    elif args.command == "train":
        train_model(args)
    else:
        raise RuntimeError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
