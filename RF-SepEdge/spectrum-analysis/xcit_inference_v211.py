#!/usr/bin/env python3
"""TorchSig 2.1.1 official narrowband clean validation-style XCiT inference.

Generates fresh synthetic IQ records using the public narrowband_clean_val_all.yaml
metadata and transforms, then runs TorchSig Models v1.0.0 XCiT (57 classes).
No training or custom signal parameters. A newly generated sample is NOT a copy
of the originally saved official validation set.

Usage:
    CUDA_VISIBLE_DEVICES=1 python xcit_inference.py --class-name qpsk
    CUDA_VISIBLE_DEVICES=1 python xcit_inference.py --samples 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsig
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.transforms import ComplexTo2D
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.yaml import load_config_from_yaml
from torchsig_models.models import XCiTClassifier

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "narrowband_clean_val_all.yaml"
DEFAULT_WEIGHTS = ROOT / "models" / "xcit_narrowband_v1.0.0.ckpt"

# Exactly the checkpoint label order used in the official GNU Radio Conference
# 2026 XCiT demo. This matches TorchSig 2.1.1's `signal_generators="all"`
# registration order (which differs from TorchSigSignalLists.all_signals).
CLASS_NAMES = [
    "tone", "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180", "ofdm-256",
    "ofdm-300", "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024",
    "ofdm-1200", "ofdm-2048", "lfm-data", "lfm-radar", "2fsk", "4fsk",
    "8fsk", "16fsk", "2gfsk", "4gfsk", "8gfsk", "16gfsk", "2msk",
    "4msk", "8msk", "16msk", "2gmsk", "4gmsk", "8gmsk", "16gmsk",
    "fm", "ook", "bpsk", "qpsk", "8psk", "16psk", "32psk", "64psk",
    "4ask", "8ask", "16ask", "32ask", "64ask", "16qam", "32qam",
    "64qam", "256qam", "1024qam", "32qam_cross", "128qam_cross",
    "512qam_cross", "chirpss", "am-dsb", "am-dsb-sc", "am-usb", "am-lsb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official narrowband 2.1.1 data -> XCiT recognition")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=None, help="Use official YAML seed by default")
    parser.add_argument("--samples", type=int, default=1, help="Number of signals to classify (default 1)")
    parser.add_argument("--class-name", choices=CLASS_NAMES, default=None,
                        help="Optional target class for demonstration; skips other random samples")
    parser.add_argument("--max-scan", type=int, default=3000,
                        help="Maximum generated samples when filtering for a target class")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def label_to_name(label) -> str:
    if isinstance(label, str):
        return label
    if isinstance(label, np.ndarray):
        if label.size != 1:
            raise ValueError(f"Expected one label; got array {label.shape}")
        return str(label.reshape(-1)[0])
    if isinstance(label, (tuple, list)):
        if len(label) != 1:
            raise ValueError(f"Expected one label; got {label!r}")
        return label_to_name(label[0])
    return str(label)


def make_loader(config: Path, seed_override: int | None):
    if torchsig.__version__ != "2.1.1":
        raise RuntimeError(f"Expected TorchSig 2.1.1, got {torchsig.__version__}")
    if not config.is_file():
        raise FileNotFoundError(f"Missing official narrowband config: {config}")
    cfg = load_config_from_yaml(config)
    if cfg.output_representation != "iq" or cfg.impairment_level != 0:
        raise ValueError("Expected official CLEAN narrowband IQ configuration")
    if cfg.signal_sampling_mode != "per_signal":
        raise ValueError("Only the official per_signal sampling is supported")

    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(cfg.dataset_metadata)
    if int(metadata["num_iq_samples_dataset"]) != 4096:
        raise ValueError("XCiT checkpoint expects 4096 IQ samples")
    if (int(metadata["num_signals_min"]), int(metadata["num_signals_max"])) != (1, 1):
        raise ValueError("XCiT narrowband classifier expects one signal per example")

    impairments = Impairments(level=cfg.impairment_level)
    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators="all",
        transforms=[impairments.dataset_transforms, ComplexTo2D()],
        component_transforms=[impairments.signal_transforms],
        target_labels=["class_name"],
    )

    # The 57-class checkpoint is indexed in generator order, NOT in the
    # separate TorchSigSignalLists.all_signals registry order.
    registered_names = list(dataset.class_names)
    if registered_names != CLASS_NAMES:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(registered_names, CLASS_NAMES)) if a != b]
        raise RuntimeError(
            "Generator class ordering differs from checkpoint mapping! "
            f"dataset count={len(registered_names)}, checkpoint count={len(CLASS_NAMES)}, "
            f"first mismatches={mismatches[:8]}"
        )

    seed = int(cfg.seed if seed_override is None else seed_override)
    loader = WorkerSeedingDataLoader(
        dataset, seed=seed, batch_size=1, num_workers=0,
        collate_fn=lambda items: items,
    )
    loader.seed(seed)
    return iter(loader), metadata, cfg, seed


def plot_iq(x: np.ndarray, label: str, pred: str, path: Path) -> None:
    n = min(512, x.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
    axes[0].plot(np.arange(n), x[0, :n], linewidth=0.8)
    axes[0].set_ylabel("I (real)")
    axes[1].plot(np.arange(n), x[1, :n], linewidth=0.8)
    axes[1].set_ylabel("Q (imag)")
    axes[1].set_xlabel("Sample index (first 512)")
    fig.suptitle(f"Ground truth: {label} | XCiT: {pred}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.max_scan < args.samples:
        raise ValueError("--samples must be positive and --max-scan >= --samples")
    print("Script version: XCIT_OFFICIAL_NARROWBAND_211_V1")
    iterator, metadata, cfg, seed = make_loader(args.config, args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = XCiTClassifier.load_from_checkpoint(
        str(args.weights), map_location="cpu", weights_only=False
    ).to(device).eval()
    num_classes = int(model.hparams.num_classes)
    if num_classes != len(CLASS_NAMES) or int(model.hparams.input_channels) != 2:
        raise RuntimeError("Checkpoint is not the expected 57-class 2-channel XCiT")

    dirname = (f"xcit_narrowband_val_{args.class_name}_seed{seed}" if args.class_name and args.samples == 1
               else f"xcit_narrowband_val_batch{args.samples}_seed{seed}" if args.samples > 1
               else f"xcit_narrowband_val_seed{seed}")
    out = (args.out or ROOT / "output" / dirname).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty: {}. Select a fresh --out.'.format(out))
    out.mkdir(parents=True, exist_ok=True)

    results = []
    scanned = 0
    while len(results) < args.samples:
        scanned += 1
        if scanned > args.max_scan:
            raise RuntimeError(f"Scanned {args.max_scan} signals but only found {len(results)} matching examples")
        raw_x, raw_label = next(iterator)[0]
        name = label_to_name(raw_label)
        if name not in CLASS_NAMES:
            raise RuntimeError(f"Unrecognized target label from v2.1.1: {name!r}")
        if args.class_name is not None and name != args.class_name:
            continue

        x = np.asarray(raw_x, dtype=np.float32)
        if x.shape != (2, 4096) or not np.isfinite(x).all():
            raise RuntimeError(f"Unexpected IQ shape or non-finite values: {x.shape}")
        batch = torch.from_numpy(x.copy()).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(batch), dim=1).squeeze(0).float().cpu().numpy()
        inds = np.argsort(-probabilities)[:5]
        pred = CLASS_NAMES[int(inds[0])]
        top5 = [{"class": CLASS_NAMES[int(k)], "probability": float(probabilities[k])}
                for k in inds]
        entry = {
            "sample_index": scanned - 1,
            "true_class": name,
            "predicted_class": pred,
            "correct": pred == name,
            "top1_confidence": float(probabilities[inds[0]]),
            "true_class_probability": float(probabilities[CLASS_NAMES.index(name)]),
            "top5": top5,
        }
        results.append(entry)
        if len(results) == 1:
            np.save(out / "input_iq.npy", (x[0] + 1j * x[1]).astype(np.complex64))
            plot_iq(x, name, pred, out / "iq_waveform.png")
        print(f"[{len(results)}/{args.samples}] true={name:<16} pred={pred:<16} "
              f"conf={entry['top1_confidence']:.4f} "
              f"{'CORRECT' if entry['correct'] else 'INCORRECT'}")

    correct = sum(r["correct"] for r in results)
    counts = Counter(r["true_class"] for r in results)
    summary = {
        "task": "single-signal narrowband IQ classification",
        "dataset": "New synthetic samples from TorchSig 2.1.1 narrowband_clean_val_all config",
        "not_original_stored_validation_records": True,
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "config_seed": int(cfg.seed),
        "seed_used": seed,
        "torchsig_version": torchsig.__version__,
        "model": args.weights.name,
        "model_class_order": CLASS_NAMES,
        "num_samples": len(results),
        "samples_scanned": scanned,
        "requested_class": args.class_name,
        "class_histogram": dict(counts),
        "correct": correct,
        "top1_accuracy": correct / len(results),
        "sample_rate_hz": float(metadata["sample_rate"]),
        "input_shape": [1, 2, 4096],
        "normalization": "No extra per-sample normalization; official ComplexTo2D result",
        "results": results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TorchSig {torchsig.__version__} | XCiT v1.0.0 | device={device}")
    print(f"Accepted {len(results)} / scanned {scanned} | Top-1 accuracy: {correct/len(results):.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
