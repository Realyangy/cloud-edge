#!/usr/bin/env python3
"""End-to-end TorchSig 2.1.1 wideband IQ -> YOLO11n -> IQ extraction -> XCiT.

A proof-of-pipeline demonstration, NOT a validated end-to-end modulation benchmark.
The YOLO model is a one-class detector. XCiT classifies an extracted, resampled
IQ segment; performance on these bursts may differ strongly from the narrowband
training distribution. Ground-truth labels are used ONLY for post-hoc evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python combined_inference.py
    CUDA_VISIBLE_DEVICES=1 python combined_inference.py --seed 1234567892
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch
import torchsig
from scipy import signal as scipy_signal
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.yaml import load_config_from_yaml
from torchsig_models.models import XCiTClassifier
from ultralytics import YOLO

# Reuse the EXACT image-conversion, box convention and matching routine that
# were validated in the existing single-sample YOLO benchmark.
from inference import (
    as_signal_list,
    box_features,
    evaluate,
    save_annotated_image,
    to_yolo_image,
    xywh_to_xyxy,
)
# This is the released XCiT checkpoint's 57-class index order, verified against
# the GRCon 2026 example. Do not replace with TorchSigSignalLists.all_signals.
from xcit_inference_v211 import CLASS_NAMES

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "wideband_clean_val_all.yaml"
YOLO_WEIGHTS = ROOT / "models" / "yolo11n_wideband_v1.0.0.pt"
# The released narrowband checkpoint is preserved at
# models/xcit_narrowband_v1.0.0.ckpt. The joint pipeline defaults to a copy
# fine-tuned on GT-box extractions from independent wideband streams.
XCIT_WEIGHTS = ROOT / "models" / "finetuned" / "xcit_joint_v3.ckpt"
NUM_XCIT_SAMPLES = 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--yolo-weights", type=Path, default=YOLO_WEIGHTS)
    p.add_argument("--xcit-weights", type=Path, default=XCIT_WEIGHTS)
    p.add_argument("--seed", type=int, default=None, help="Defaults to official wideband YAML seed")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5, help="GT matching threshold for evaluation only")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | 0 (GPU index after CUDA_VISIBLE_DEVICES)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def make_joint_sample(config_path: Path, seed_override: int | None):
    """Generate ONE sample, capturing pre-spectrogram IQ without modifying it.

    Adding a read-only capture transform between dataset impairment and
    Spectrogram ensures the spectrogram and IQ really are from the SAME sample.
    Generator, transforms and labels otherwise match inference.py exactly.
    """
    if torchsig.__version__ != "2.1.1":
        raise RuntimeError(f"Expected torchsig==2.1.1; found {torchsig.__version__}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing official config: {config_path}")
    cfg = load_config_from_yaml(config_path)
    if (cfg.output_representation, cfg.impairment_level, cfg.signal_sampling_mode) != (
        "spectrogram", 0, "per_signal"
    ):
        raise ValueError("Expected official CLEAN per-signal wideband spectrogram configuration")
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(cfg.dataset_metadata)
    seed = int(cfg.seed if seed_override is None else seed_override)
    fft_size = int(metadata["fft_size"])
    captures: dict[str, np.ndarray] = {}

    def save_iq(signal_obj):
        arr = np.asarray(signal_obj.data)
        if not np.iscomplexobj(arr) or len(arr) != int(metadata["num_iq_samples_dataset"]):
            raise RuntimeError(f"Unexpected raw IQ before STFT: {arr.dtype}, {arr.shape}")
        captures["iq"] = np.asarray(arr, dtype=np.complex64).copy()
        return signal_obj  # read-only: must not perturb model's input

    impairments = Impairments(level=cfg.impairment_level)
    ds = TorchSigIterableDataset(
        signal_generators="all",
        metadata=metadata,
        transforms=[impairments.dataset_transforms, save_iq,
                    Spectrogram(fft_size=fft_size), YOLOLabel()],
        component_transforms=[impairments.signal_transforms],
        target_labels=["yolo_label", "class_name"],
    )
    dl = WorkerSeedingDataLoader(
        ds, seed=seed, batch_size=1, num_workers=0, collate_fn=lambda b: b
    )
    dl.seed(seed)
    spectrogram, targets = next(iter(dl))[0]
    if "iq" not in captures:
        raise RuntimeError("Raw IQ capture callback was not invoked")
    spec = np.asarray(spectrogram, dtype=np.float32)
    if spec.shape != (fft_size, fft_size):
        raise RuntimeError(f"Unexpected spectrogram shape: {spec.shape}")
    gt_boxes = as_signal_list(targets[0], "box")
    gt_names = [str(n) for n in as_signal_list(targets[1], "name")]
    if len(gt_boxes) != len(gt_names):
        raise RuntimeError("Mismatch between GT boxes and GT class labels")
    return captures["iq"], spec, gt_boxes, gt_names, metadata, cfg, seed


def signal_family(name: str) -> str:
    if name.startswith("ofdm-"):
        return "OFDM"
    if "qam" in name:
        return "QAM"
    if "psk" in name:
        return "PSK"
    if name.endswith(("gmsk", "msk")):
        return "MSK/GMSK"
    if name.endswith(("gfsk", "fsk")):
        return "FSK/GFSK"
    if "ask" in name or name == "ook":
        return "ASK/OOK"
    if name.startswith("am-"):
        return "AM"
    if name == "fm":
        return "FM"
    return "Other"


def extract_iq(raw: np.ndarray, box: list[float], fs: float) -> tuple[np.ndarray, dict]:
    """Predicted normalized box -> baseband IQ [2,4096], using no true labels.

    Bandlimited downconversion and polyphase resampling are engineering
    approximations, not an exact reconstruction of XCiT's training distribution.
    We keep enough temporal samples when possible, and report padding/ratios.
    """
    x0, y0, x1, y1 = [float(t) for t in box]
    n = len(raw)
    s = max(0, min(n, int(np.floor(x0 * n))))
    e = max(0, min(n, int(np.ceil(x1 * n))))
    if e - s < 256:
        raise ValueError(f"Detected time segment too short: {e-s} IQ samples")
    center_hz = (0.5 - 0.5 * (y0 + y1)) * fs
    bw_hz = (y1 - y0) * fs
    if bw_hz <= 0 or bw_hz >= fs:
        raise ValueError(f"Invalid predicted bandwidth: {bw_hz} Hz")
    segment = np.asarray(raw[s:e], dtype=np.complex64)
    local_t = np.arange(len(segment), dtype=np.float64) / fs
    baseband = segment * np.exp(-2j * np.pi * center_hz * local_t)

    # Preserve slightly more than predicted half-bandwidth, but eliminate
    # out-of-band energy before reducing the sampling rate.
    lowpass_cutoff_hz = min(bw_hz * 0.60, fs * 0.47)
    taps = scipy_signal.firwin(129, lowpass_cutoff_hz, fs=fs, window="hamming")
    baseband = scipy_signal.fftconvolve(baseband, taps, mode="same")
    duration_s = len(segment) / fs
    # Target relative occupied bandwidth ~0.30, similar to the narrowband
    # dataset. Increase output rate when needed for 4096 temporal samples.
    desired_rate = 3.0 * bw_hz
    enough_samples_rate = NUM_XCIT_SAMPLES * 1.02 / duration_s
    output_rate = min(fs, max(desired_rate, enough_samples_rate))
    rate_ratio = Fraction(float(output_rate / fs)).limit_denominator(4096)
    up, down = rate_ratio.numerator, rate_ratio.denominator
    resampled = scipy_signal.resample_poly(baseband, up, down).astype(np.complex64)
    num_resampled = len(resampled)
    if num_resampled >= NUM_XCIT_SAMPLES:
        center = num_resampled // 2
        first = center - NUM_XCIT_SAMPLES // 2
        out = resampled[first:first + NUM_XCIT_SAMPLES]
        pad_count = 0
    else:
        pad_count = NUM_XCIT_SAMPLES - num_resampled
        out = np.pad(resampled, (pad_count // 2, pad_count - pad_count // 2))
    # Complex IQ RMS normalization after resampling and cropping
    rms = float(np.sqrt(np.mean(np.abs(out.astype(np.complex128)) ** 2)))
    if not np.isfinite(rms) or rms < 1e-8:
        raise ValueError(f"Invalid IQ RMS: {rms}")
    out = (out / rms).astype(np.complex64)
    x = np.stack((out.real, out.imag), axis=0).astype(np.float32)
    if x.shape != (2, NUM_XCIT_SAMPLES) or not np.isfinite(x).all():
        raise RuntimeError("Invalid channelized XCiT input")
    actual_fs = fs * up / down
    quality = {
        "normalization": "complex_rms",
        "rms_before_normalization": rms,
        "start_sample": s,
        "stop_sample_exclusive": e,
        "input_iq_count": e - s,
        "predicted_center_offset_hz": round(center_hz, 3),
        "predicted_bandwidth_hz": round(bw_hz, 3),
        "lowpass_cutoff_hz": round(lowpass_cutoff_hz, 3),
        "resample_up": int(up),
        "resample_down": int(down),
        "estimated_output_rate_hz": round(actual_fs, 3),
        "bandwidth_to_output_rate_ratio": round(bw_hz / actual_fs, 4),
        "pre_crop_samples": num_resampled,
        "zero_padding_samples": int(pad_count),
        "caveat": "Classification input may differ from official clean narrowband training distribution",
    }
    return x, quality


def classify_iq(model, x: np.ndarray, device: str) -> dict:
    with torch.inference_mode():
        batch = torch.from_numpy(x).unsqueeze(0).to(device)
        probabilities = model(batch).softmax(dim=1).squeeze(0).float().cpu().numpy()
    if len(probabilities) != len(CLASS_NAMES):
        raise RuntimeError(f"Incorrect XCiT output width: {len(probabilities)}")
    ranked = np.argsort(-probabilities)
    top = int(ranked[0])
    fam_sums: dict[str, float] = {}
    for label, p in zip(CLASS_NAMES, probabilities):
        f = signal_family(label)
        fam_sums[f] = fam_sums.get(f, 0.0) + float(p)
    families = sorted(fam_sums.items(), key=lambda pair: pair[1], reverse=True)
    return {
        "fine_top1": CLASS_NAMES[top],
        "fine_top1_confidence": round(float(probabilities[top]), 6),
        "fine_top5": [{"class": CLASS_NAMES[int(j)],
                       "probability": round(float(probabilities[j]), 6)} for j in ranked[:5]],
        "family_from_fine_top1": signal_family(CLASS_NAMES[top]),
        "family_top1": families[0][0],
        "family_top1_probability_sum": round(families[0][1], 6),
        "family_probabilities": {key: round(value, 6) for key, value in families},
    }


def draw_overlay(image: np.ndarray, preds: list[dict], dest: Path) -> None:
    overlay = image.copy()
    h, w = overlay.shape[:2]
    for i, record in enumerate(preds, start=1):
        x0, y0, x1, y1 = record["box_xyxy_normalized"]
        p0 = (int(np.clip(x0 * w, 0, w-1)), int(np.clip(y0 * h, 0, h-1)))
        p1 = (int(np.clip(x1 * w, 0, w-1)), int(np.clip(y1 * h, 0, h-1)))
        cv2.rectangle(overlay, p0, p1, (0, 125, 250), 2)
        result = record.get("xcit", {})
        fam = result.get("family_top1", "unavailable")
        label = f"{i}: {fam} ({record['confidence']:.2f})"
        yy = max(14, p0[1] - 5)
        cv2.putText(overlay, label, (p0[0], yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (15, 60, 250), 1, cv2.LINE_AA)
    cv2.imwrite(str(dest), overlay)


def main() -> None:
    args = parse_args()
    if not (0 < args.conf < 1 and 0 < args.iou <= 1):
        raise ValueError("--conf and --iou must be within (0,1)")
    raw, spec, gt_labels, gt_names, meta, cfg, seed = make_joint_sample(args.config, args.seed)
    out = (args.out or ROOT / "output" / f"combined_seed{seed}").resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty: {}. Select a fresh --out.'.format(out))
    out.mkdir(parents=True, exist_ok=True)

    fs = float(meta["sample_rate"])
    duration_s = len(raw) / fs
    image = to_yolo_image(spec)
    src = out / "spectrogram.png"
    if not cv2.imwrite(str(src), image):
        raise RuntimeError(f"Cannot save image: {src}")
    np.save(out / "wideband_iq.npy", raw)

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    classification_device = "cuda:0" if device == "0" else device
    if device.isdigit() and device != "0":
        classification_device = f"cuda:{device}"

    yolo = YOLO(str(args.yolo_weights))
    box_result = yolo.predict(source=str(src), imgsz=image.shape[0],
                              conf=args.conf, device=device, verbose=False)[0]
    classifier = XCiTClassifier.load_from_checkpoint(
        str(args.xcit_weights), map_location="cpu", weights_only=False
    ).to(classification_device).eval()
    if int(classifier.hparams.num_classes) != len(CLASS_NAMES) or int(classifier.hparams.input_channels) != 2:
        raise RuntimeError("Unexpected XCiT checkpoint dimensions")

    gt = []
    for cls, label in zip(gt_names, gt_labels):
        _, cx, cy, bw, bh = map(float, label)
        box = xywh_to_xyxy(cx, cy, bw, bh)
        gt.append({"true_class": cls, "box_xyxy_normalized": box,
                   "features": box_features(box, fs, duration_s)})

    preds = []
    h, w = image.shape[:2]
    for i, (boxpix, conf) in enumerate(zip(box_result.boxes.xyxy.cpu().numpy(),
                                            box_result.boxes.conf.cpu().numpy()), start=1):
        a, b, c, d = map(float, boxpix)
        box = [a/w, b/h, c/w, d/h]
        record = {
            "detection_id": i,
            "predicted_class": "signal",
            "confidence": round(float(conf), 6),
            "box_xyxy_normalized": box,
            "features": box_features(box, fs, duration_s),
        }
        try:
            extracted, quality = extract_iq(raw, box, fs)
            np.save(out / f"signal_{i:02d}_iq.npy", extracted)
            record["extraction"] = quality
            record["xcit"] = classify_iq(classifier, extracted, classification_device)
        except (ValueError, RuntimeError) as exc:
            record["xcit_error"] = str(exc)
        preds.append(record)

    # Ground-truth names are used ONLY HERE, after detection and classification.
    metrics = evaluate(preds, gt, args.iou)
    for match in metrics["matches"]:
        pred = preds[match["prediction_index"]]
        ground = gt[match["ground_truth_index"]]
        if "xcit" not in pred:
            continue
        xcit = pred["xcit"]
        xcit["matched_true_class"] = ground["true_class"]
        xcit["matched_true_family"] = signal_family(ground["true_class"])
        xcit["fine_correct"] = (xcit["fine_top1"] == ground["true_class"])
        xcit["family_correct"] = (xcit["family_top1"] == signal_family(ground["true_class"]))

    matched = [p["xcit"] for p in preds if "xcit" in p and "matched_true_class" in p["xcit"]]
    metrics["matched_classified"] = len(matched)
    metrics["matched_fine_correct"] = sum(int(x["fine_correct"]) for x in matched)
    metrics["matched_family_correct"] = sum(int(x["family_correct"]) for x in matched)
    metrics["matched_fine_accuracy"] = (metrics["matched_fine_correct"] / len(matched)) if matched else None
    metrics["matched_family_accuracy"] = (metrics["matched_family_correct"] / len(matched)) if matched else None
    save_annotated_image(image, gt, preds, fs, duration_s, out / "detections.png")
    draw_overlay(image, preds, out / "combined_overlay.png")

    result = {
        "task": "Wideband signal detection -> channelized IQ -> fine/family modulation recognition",
        "status": "research prototype; joint XCiT classification not validated",
        "model_detection": args.yolo_weights.name,
        "model_recognition": args.xcit_weights.name,
        "torchsig_version": torchsig.__version__,
        "dataset_config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "seed": seed,
        "sample_rate_hz": fs,
        "duration_ms": 1000 * duration_s,
        "raw_iq_shape": list(raw.shape),
        "spectrogram_shape": list(spec.shape),
        "note": "Same sample for YOLO and XCiT. XCiT uses PREDICTED boxes only. True labels used for evaluation only.",
        "limitations": [
            "YOLO predicts a single 'signal' detection class; modulation comes from XCiT only.",
            "Band-limited burst extraction and resampling may cause a significant distribution shift.",
            "A single synthetic sample cannot establish end-to-end classification accuracy.",
            "Narrowband XCiT training bandwidth and burst-duration distributions differ from wideband data.",
        ],
        "ground_truth": gt,
        "detections": preds,
        "metrics": metrics,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TorchSig {torchsig.__version__} | YOLO11n + XCiT | device={device}")
    print(f"Raw IQ: {len(raw)} complex samples | GT signals={len(gt)} | predicted detections={len(preds)}")
    for p in preds:
        f = p["features"]
        xc = p.get("xcit", {})
        fcode = xc.get("fine_top1", "N/A")
        fam = xc.get("family_top1", "N/A")
        truth = xc.get("matched_true_class", "unmatched")
        print(f"  #{p['detection_id']}: f={f['center_frequency_mhz_offset']:+.3f}MHz "
              f"BW={f['bandwidth_mhz']:.3f}MHz YOLO={p['confidence']:.3f} "
              f"XCiT={fcode} family={fam} GT={truth}")
    print(f"YOLO TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']} | "
          f"matched family correct={metrics['matched_family_correct']}/{metrics['matched_classified']}")
    print("Saved:", out)


if __name__ == "__main__":
    main()
