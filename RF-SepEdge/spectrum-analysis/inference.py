#!/usr/bin/env python3
"""One-sample TorchSig 2.1.1 official wideband CLEAN validation configuration + YOLO11n.

Generates a *new synthetic sample* from the published validation *configuration*;
this is not a stored, independently held-out validation example. The YOLO model
is a single-class signal detector; it does not identify modulation type.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torchsig
from ultralytics import YOLO

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.yaml import load_config_from_yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "wideband_clean_val_all.yaml"
DEFAULT_WEIGHTS = ROOT / "models" / "yolo11n_wideband_v1.0.0.pt"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one TorchSig 2.1.1 official wideband clean validation-style sample and run YOLO11n"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=None, help="If omitted, use seed from official YAML")
    parser.add_argument("--out", type=Path, default=None,
                        help="If omitted, output/yolo11n_clean_val_seed<seed>")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold; default 0.25")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for sample-level ground-truth matching")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cpu, 0 etc. CUDA_VISIBLE_DEVICES remaps indices")
    return parser.parse_args()


def as_signal_list(value, kind: str) -> list:
    """Normalize a TorchSig scalar/singleton target to a list of per-signal values."""
    if value is None:
        return []
    if kind == "name":
        if isinstance(value, str):
            return [value]
        return list(value)
    # One YOLO tuple: (cls, center_x, center_y, width, height)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 5 and np.isscalar(value[0]):
        return [value]
    return list(value)


def generate_sample(config_path: Path, seed_override: int | None):
    """Mirror TorchSig v2.1.1 scripts/generate_dataset_from_config.py with one sample."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Official config YAML not found: {config_path}")
    cfg = load_config_from_yaml(config_path)
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(cfg.dataset_metadata)

    if cfg.output_representation != "spectrogram":
        raise ValueError("Official YAML must specify spectrogram output")
    if cfg.signal_sampling_mode != "per_signal":
        raise ValueError("This demo implements per_signal sampling only")
    if cfg.impairment_level != 0:
        raise ValueError("This demo expects the CLEAN validation configuration (impairment_level=0)")
    if not str(torchsig.__version__).startswith("2.1.1"):
        raise RuntimeError(f"Expected torchsig==2.1.1, found {torchsig.__version__}")

    seed = int(cfg.seed if seed_override is None else seed_override)
    fft_size = int(metadata["fft_size"])
    impairments = Impairments(level=cfg.impairment_level)
    # Match the public v2.1.1 official dataset generation script:
    # whole-sample impairments -> spectrogram -> YOLO label; plus component transforms.
    dataset = TorchSigIterableDataset(
        signal_generators="all",
        metadata=metadata,
        transforms=[impairments.dataset_transforms, Spectrogram(fft_size=fft_size), YOLOLabel()],
        component_transforms=[impairments.signal_transforms],
        target_labels=["yolo_label", "class_name"],
    )
    loader = WorkerSeedingDataLoader(
        dataset, seed=seed, batch_size=1, num_workers=0,
        collate_fn=lambda batch: batch,
    )
    loader.seed(seed)
    spec, targets = next(iter(loader))[0]
    gt_yolo = as_signal_list(targets[0], "box")
    gt_names = as_signal_list(targets[1], "name")
    spec = np.asarray(spec, dtype=np.float32)
    if spec.shape != (fft_size, fft_size):
        raise RuntimeError(f"Expected ({fft_size}, {fft_size}) spectrogram, got {spec.shape}")
    if len(gt_names) != len(gt_yolo):
        raise RuntimeError(f"Label count mismatch: {len(gt_names)} names versus {len(gt_yolo)} boxes")
    return spec, gt_yolo, [str(n) for n in gt_names], metadata, cfg, seed


def to_yolo_image(spec: np.ndarray) -> np.ndarray:
    """Match torchsig_models.adapters.yolo_utils.static_to_yolo() exactly."""
    h, w = spec.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    image = cv2.normalize(spec, image, 0, 255, cv2.NORM_MINMAX)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_not(image)


def xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]


def box_features(box: list[float], sample_rate: float, duration_s: float) -> dict:
    x0, y0, x1, y1 = box
    upper_hz = (0.5 - y0) * sample_rate
    lower_hz = (0.5 - y1) * sample_rate
    return {
        "start_time_ms": round(x0 * duration_s * 1000, 6),
        "end_time_ms": round(x1 * duration_s * 1000, 6),
        "duration_ms": round((x1 - x0) * duration_s * 1000, 6),
        "frequency_lower_mhz_offset": round(lower_hz / 1e6, 6),
        "frequency_upper_mhz_offset": round(upper_hz / 1e6, 6),
        "center_frequency_mhz_offset": round((upper_hz + lower_hz) / 2e6, 6),
        "bandwidth_mhz": round((upper_hz - lower_hz) / 1e6, 6),
    }


def box_iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(preds: list[dict], truth: list[dict], threshold: float) -> dict:
    used, matches = set(), []
    for p_idx, pred in enumerate(preds):
        candidates = [
            (box_iou(pred["box_xyxy_normalized"], gt["box_xyxy_normalized"]), g_idx)
            for g_idx, gt in enumerate(truth) if g_idx not in used
        ]
        if not candidates:
            continue
        overlap, g_idx = max(candidates)
        if overlap >= threshold:
            used.add(g_idx)
            matches.append({"prediction_index": p_idx, "ground_truth_index": g_idx,
                            "iou": round(float(overlap), 6)})
    tp = len(matches)
    return {
        "match_iou_threshold": threshold,
        "TP": tp, "FP": len(preds) - tp, "FN": len(truth) - tp,
        "precision": tp / len(preds) if preds else 0.0,
        "recall": tp / len(truth) if truth else 0.0,
        "mean_matched_iou": round(sum(m["iou"] for m in matches) / tp, 6) if tp else None,
        "matches": matches,
    }


def save_annotated_image(image: np.ndarray, ground_truth: list[dict], preds: list[dict],
                         sample_rate: float, duration_s: float, filename: Path) -> None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    extent = [0, duration_s * 1000, -sample_rate / 2e6, sample_rate / 2e6]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, items, color, title in (
        (axes[0], ground_truth, "#168fb5", "Ground truth (TorchSig labels)"),
        (axes[1], preds, "#ef8b42", "YOLO11n detections (class: signal)"),
    ):
        ax.imshow(image_rgb, origin="upper", extent=extent, interpolation="nearest", aspect="auto")
        ax.set(xlabel="Time (ms)", ylabel="Frequency offset (MHz)", title=title)
        for item in items:
            f = item["features"]
            ax.add_patch(Rectangle((f["start_time_ms"], f["frequency_lower_mhz_offset"]),
                                   f["duration_ms"], f["bandwidth_mhz"],
                                   edgecolor=color, linewidth=1.7, fill=False))
            text = item.get("true_class", "signal")
            if "confidence" in item:
                text += f" {item['confidence']:.3f}"
            ax.text(f["start_time_ms"], f["frequency_upper_mhz_offset"], text,
                    fontsize=8, color="white", va="bottom",
                    bbox={"facecolor": color, "alpha": 0.85, "pad": 2})
    fig.savefig(filename, dpi=145)
    plt.close(fig)


def main() -> None:
    args = arguments()
    if not (0.0 <= args.conf <= 1.0 and 0.0 <= args.iou <= 1.0):
        raise ValueError("--conf and --iou must be within [0,1]")

    spec, labels, names, metadata, cfg, seed = generate_sample(args.config, args.seed)
    out_dir = (args.out or ROOT / "output" / f"yolo11n_clean_val_seed{seed}").resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}; choose a new --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = to_yolo_image(spec)
    source_path = out_dir / "spectrogram.png"
    if not cv2.imwrite(str(source_path), img):
        raise RuntimeError(f"Failed to write {source_path}")

    device = "0" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device
    model = YOLO(str(args.weights))
    predictions_raw = model.predict(source=str(source_path), imgsz=img.shape[0],
                                    conf=args.conf, device=device, verbose=False)[0]
    h, w = img.shape[:2]
    sample_rate = float(metadata["sample_rate"])
    duration_s = float(metadata["num_iq_samples_dataset"]) / sample_rate

    truth = []
    for name, label in zip(names, labels):
        _, cx, cy, bw, bh = map(float, label)
        box = xywh_to_xyxy(cx, cy, bw, bh)
        truth.append({"true_class": name, "box_xyxy_normalized": box,
                      "features": box_features(box, sample_rate, duration_s)})

    preds = []
    for bounds, score in zip(predictions_raw.boxes.xyxy.cpu().numpy(),
                             predictions_raw.boxes.conf.cpu().numpy()):
        x0, y0, x1, y1 = map(float, bounds)
        box = [x0 / w, y0 / h, x1 / w, y1 / h]
        preds.append({"predicted_class": "signal", "confidence": round(float(score), 6),
                      "box_xyxy_normalized": box,
                      "features": box_features(box, sample_rate, duration_s)})

    metrics = evaluate(preds, truth, args.iou)
    save_annotated_image(img, truth, preds, sample_rate, duration_s, out_dir / "detections.png")
    result = {
        "task": "wideband RF signal detection and time-frequency feature extraction",
        "dataset": "TorchSig 2.1.1 official wideband_clean_val_all configuration; newly generated sample",
        "benchmark_caveat": "Generated with official validation parameters, but not claimed to be an identical held-out official validation record.",
        "torchsig_version": torchsig.__version__,
        "model": args.weights.name,
        "model_class_names": model.names,
        "model_note": "YOLO11n is a single-class detector; ground-truth modulation names are not YOLO predictions.",
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "config_dataset_id": cfg.dataset_id,
        "config_seed": int(cfg.seed),
        "seed_used": seed,
        "impairment_level": int(cfg.impairment_level),
        "configured_signal_count_range": [int(metadata["num_signals_min"]), int(metadata["num_signals_max"])],
        "realized_signal_count": len(truth),
        "sample_rate_hz": sample_rate,
        "capture_duration_ms": round(duration_s * 1000, 6),
        "spectrogram_size": [h, w],
        "frequency_reference": "baseband offsets; absolute carrier frequency unspecified",
        "confidence_threshold": args.conf,
        "ground_truth": truth,
        "detections": preds,
        "single_sample_metrics": metrics,
    }
    (out_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"TorchSig {torchsig.__version__} | dataset={cfg.dataset_id} | YOLO11n | device={device}")
    print(f"Config seed: {cfg.seed}; used seed: {seed}; configured signals: {metadata['num_signals_min']}-{metadata['num_signals_max']}")
    print(f"Ground truth: {len(truth)} signals; detections: {len(preds)}")
    for i, gt in enumerate(truth, 1):
        f = gt["features"]
        print(f"  GT {i}: {gt['true_class']} center={f['center_frequency_mhz_offset']:+.3f} MHz BW={f['bandwidth_mhz']:.3f} MHz")
    for i, pred in enumerate(preds, 1):
        f = pred["features"]
        print(f"  Pred {i}: signal conf={pred['confidence']:.4f} center={f['center_frequency_mhz_offset']:+.3f} MHz BW={f['bandwidth_mhz']:.3f} MHz")
    print(f"Single-sample match: TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, mean IoU={metrics['mean_matched_iou']}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
