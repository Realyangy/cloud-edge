from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
import torchsig
import yaml
from ultralytics import YOLO

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.yaml import load_config_from_yaml

# Reuse the known-good official image conversion and target-list normalization.
from inference import as_signal_list, to_yolo_image

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "wideband_clean_val_all.yaml"
DEFAULT_WEIGHTS = HERE / "models" / "yolo11n_wideband_v1.0.0.pt"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50, help="Number of synthetic validation-style examples")
    parser.add_argument("--seed", type=int, default=None, help="Override official configuration seed")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=None, help="Output root; defaults to output/yolo11n_batch<N>_seed<S>")
    parser.add_argument("--batch", type=int, default=8, help="YOLO validation batch size")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, 0; CUDA_VISIBLE_DEVICES remaps device indices")
    return parser.parse_args()


def generate_samples(args, output_dir, cfg, seed):
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(cfg.dataset_metadata)
    fft_size = int(metadata["fft_size"])
    impairments = Impairments(level=cfg.impairment_level)
    # Same pipeline as TorchSig v2.1.1 scripts/generate_dataset_from_config.py
    dataset = TorchSigIterableDataset(
        signal_generators="all",
        metadata=metadata,
        transforms=[impairments.dataset_transforms, Spectrogram(fft_size=fft_size), YOLOLabel()],
        component_transforms=[impairments.signal_transforms],
        target_labels=["yolo_label", "class_name"],
    )
    loader = WorkerSeedingDataLoader(
        dataset, seed=seed, batch_size=1, num_workers=0, collate_fn=lambda batch: batch
    )
    loader.seed(seed)
    stream = iter(loader)

    image_dir = output_dir / "yolo_dataset" / "images" / "val"
    label_dir = output_dir / "yolo_dataset" / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    sample_metadata = []
    total_objects = 0
    for i in range(args.samples):
        spec, targets = next(stream)[0]
        boxes = as_signal_list(targets[0], "box")
        names = as_signal_list(targets[1], "name")
        if len(boxes) != len(names):
            raise RuntimeError(f"Sample {i}: target mismatch: {len(boxes)} boxes / {len(names)} names")
        spec = np.asarray(spec, dtype=np.float32)
        if spec.shape != (fft_size, fft_size):
            raise RuntimeError(f"Sample {i}: invalid spectrogram shape {spec.shape}")
        bgr = to_yolo_image(spec)
        stem = f"{i:06d}"
        img_path = image_dir / f"{stem}.png"
        if not cv2.imwrite(str(img_path), bgr):
            raise RuntimeError(f"Failed saving {img_path}")
        label_lines = []
        truth = []
        for raw_box, true_name in zip(boxes, names):
            cls, cx, cy, w, h = map(float, raw_box)
            # The released model is *single class*, unlike the TorchSig 57-class metadata.
            if not np.isfinite([cx, cy, w, h]).all():
                raise RuntimeError(f"Sample {i}: nonfinite label {raw_box}")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise RuntimeError(f"Sample {i}: out-of-bounds YOLO label {raw_box}")
            label_lines.append(f"0 {cx:.9f} {cy:.9f} {w:.9f} {h:.9f}")
            truth.append({"true_class": str(true_name), "class_index_original": int(cls),
                          "box_cxcywh": [cx, cy, w, h]})
        (label_dir / f"{stem}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        sample_metadata.append({"sample_index": i, "image": str(img_path.relative_to(output_dir)),
                                "signal_count": len(truth), "ground_truth": truth})
        total_objects += len(truth)
        if (i + 1) % 10 == 0 or i + 1 == args.samples:
            print(f"Prepared {i+1}/{args.samples} images; {total_objects} total signals", flush=True)

    data_yaml = output_dir / "yolo_dataset" / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(data_yaml.parent.resolve()),
        "train": "images/val",
        "val": "images/val",
        "nc": 1,
        "names": {0: "signal"},
    }, sort_keys=False), encoding="utf-8")
    (output_dir / "samples_metadata.json").write_text(
        json.dumps(sample_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data_yaml, total_objects


def main():
    args = get_args()
    if not str(torchsig.__version__).startswith("2.1.1"):
        raise RuntimeError(f"Expected torchsig 2.1.1, got {torchsig.__version__}")
    if args.samples < 1 or args.batch < 1:
        raise ValueError("--samples and --batch must be positive integers")
    cfg = load_config_from_yaml(args.config)
    if cfg.output_representation != "spectrogram" or cfg.signal_sampling_mode != "per_signal":
        raise ValueError("Expected per_signal spectrogram validation configuration")
    if cfg.impairment_level != 0 or cfg.dataset_id != "wideband_clean_val_all":
        raise ValueError("Expected official wideband_clean_val_all (impairment_level 0)")
    seed = int(cfg.seed if args.seed is None else args.seed)
    output_dir = (args.out or HERE / "output" / f"yolo11n_batch{args.samples}_seed{seed}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; select a new --out")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "source_config.yaml")
    data_yaml, total_objects = generate_samples(args, output_dir, cfg, seed)

    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Validating {args.samples} images / {total_objects} signals on device={device}...", flush=True)
    model = YOLO(str(args.weights))
    # AP evaluation uses low threshold and aggregates PR across the entire set.
    metrics = model.val(
        data=str(data_yaml), split="val", imgsz=int(cfg.dataset_metadata["fft_size"]),
        batch=args.batch, device=device, conf=0.001, iou=0.7,
        workers=0, single_cls=True, verbose=False,
        plots=True, project=str(output_dir), name="validation", exist_ok=True,
    )
    values = {
        "dataset_provenance": "Regenerated TorchSig 2.1.1 validation-configuration synthetic samples, NOT official held-out stored records",
        "torchsig_version": str(torchsig.__version__),
        "model": args.weights.name,
        "config_dataset_id": cfg.dataset_id,
        "seed": seed,
        "sample_count": args.samples,
        "true_signal_count": total_objects,
        "detector_class": "signal (single class)",
        "IoU_0_5_precision_from_PR_curve": float(metrics.box.mp),
        "IoU_0_5_recall_from_PR_curve": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "mAP50_95": float(metrics.box.map),
        "notes": "PR-derived precision/recall may differ from the fixed conf=0.25 per-image counts in inference.py.",
    }
    (output_dir / "summary.json").write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== YOLO11n {args.samples}-sample validation-style results ===")
    print(f"Images: {args.samples}; GT signal instances: {total_objects}")
    for key in ["IoU_0_5_precision_from_PR_curve", "IoU_0_5_recall_from_PR_curve", "mAP50", "mAP75", "mAP50_95"]:
        print(f"{key}: {values[key]:.4f}")
    print(f"Saved summary and dataset in: {output_dir}")


if __name__ == "__main__":
    main()
