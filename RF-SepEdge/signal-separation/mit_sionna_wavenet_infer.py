#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run official MIT WaveNet checkpoints on SMALL Sionna-generated QPSK/OFDMQPSK examples.

Step 1: Generate data in rf-sionna using mit_sionna_official_smoke.py.
Step 2: Run THIS file in rf-spectrum:
    CUDA_VISIBLE_DEVICES=1 python mit_sionna_wavenet_infer.py

This script uses the original test-mixture and ground-truth SOI arrays generated
by the MIT v0.2.0 Sionna routine. It does not use true SOI during inference.
It reports only waveform MSE/NMSE and saves predictions for subsequent OFFICIAL
Sionna QPSK/OFDM demodulation. It is NOT the full MIT 1100-sample evaluation.

Safety: The official .pt checkpoint includes serialized OmegaConf objects, so
weights_only=False is necessary. Only use verified, trusted checkpoint files.
The locally generated groundtruth .pkl is also a pickle file; never open one
from an untrusted source.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SOI_TYPES = ("QPSK", "OFDMQPSK")
INTERFERENCE = "CommSignal2"
SAMPLE_SIZE = 40960


def pwr(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.complex128)
    return float(np.mean(np.abs(v) ** 2))


def stats(ref: np.ndarray, estimate: np.ndarray) -> dict:
    if ref.shape != estimate.shape:
        raise ValueError("shape mismatch: {} vs {}".format(ref.shape, estimate.shape))
    err = np.asarray(estimate, dtype=np.complex128) - np.asarray(ref, dtype=np.complex128)
    per_record_mse = np.mean(np.abs(err) ** 2, axis=1)
    per_record_ref_power = np.mean(np.abs(np.asarray(ref, dtype=np.complex128)) ** 2, axis=1)
    per_record_nmse_db = 10 * np.log10(np.maximum(per_record_mse, 1e-30) / np.maximum(per_record_ref_power, 1e-30))
    return {
        "mse": per_record_mse.tolist(),
        "mse_db": (10 * np.log10(np.maximum(per_record_mse, 1e-30))).tolist(),
        "nmse_db": per_record_nmse_db.tolist(),
        "mean_mse_db": float(10 * np.log10(max(float(np.mean(per_record_mse)), 1e-30))),
        "mean_nmse_db": float(np.mean(per_record_nmse_db)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source", type=Path, default=Path("output/mit_sionna_official_smoke"))
    parser.add_argument("--out", type=Path, default=Path("output/mit_sionna_wavenet_infer"))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError("--batch-size must be 1..8 for smoke test")
    root = args.root.resolve()
    source = (args.source if args.source.is_absolute() else root / args.source).resolve()
    out = (args.out if args.out.is_absolute() else root / args.out).resolve()
    src_dir = root / "rfchallenge"
    config_file = src_dir / "src/configs/wavenet.yml"
    manifest_file = source / "summary.json"
    if not config_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("Expected config and Sionna summary.json; check paths: {} {}".format(config_file, manifest_file))

    with manifest_file.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("interference") != INTERFERENCE:
        raise ValueError("This script expects CommSignal2; got {}".format(manifest.get("interference")))
    if sorted([r["soi_type"] for r in manifest.get("results", [])]) != sorted(SOI_TYPES):
        raise ValueError("Expected Sionna QPSK and OFDMQPSK source files in manifest")

    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(src_dir))
    from src.torchwavenet import Wave
    from src.config_torchwavenet import ModelConfig

    raw_model_cfg = OmegaConf.load(str(config_file))
    cfg = ModelConfig(**OmegaConf.to_container(raw_model_cfg.model, resolve=True))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=== Official Sionna mixtures -> official WaveNet ===", flush=True)
    print("Device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)
    print("Sionna input folder:", source, flush=True)
    print("Inference output folder:", out, flush=True)
    print("Generator:", manifest.get("source"), flush=True)
    print("Sample size:", SAMPLE_SIZE, flush=True)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty: {}. Select a fresh --out.'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "WaveNet inference on official MIT Sionna-generated MINI-set, not complete official benchmark",
        "input_sionna_summary": str(manifest_file),
        "models": [],
    }

    for soi in SOI_TYPES:
        stem = "TestSet1Example_testmixture_{}_{}".format(soi, INTERFERENCE)
        mix_path = source / "dataset" / (stem + ".npy")
        truth_path = source / "dataset" / "GroundTruth_TestSet1Example_Dataset_{}_{}.pkl".format(soi, INTERFERENCE)
        ckpt_path = root / "pretrained-models" / "WaveNet" / "dataset_{}_{}_mixture_wavenet".format(soi.lower(), INTERFERENCE.lower()) / "weights.pt"
        for path in (mix_path, truth_path, ckpt_path):
            if not path.is_file():
                raise FileNotFoundError(str(path))
        mixtures = np.load(str(mix_path), allow_pickle=False)
        # Generated locally by original MIT Sionna function; trusted local file only.
        with truth_path.open("rb") as fp:
            truth_mixtures, target, bits = pickle.load(fp)
        mixtures = np.asarray(mixtures, dtype=np.complex64)
        target = np.asarray(target, dtype=np.complex64)
        bits = np.asarray(bits)
        if mixtures.ndim != 2 or mixtures.shape[1] != SAMPLE_SIZE:
            raise ValueError("Unexpected mixture dimensions {}".format(mixtures.shape))
        if mixtures.shape != target.shape or mixtures.shape != np.asarray(truth_mixtures).shape:
            raise ValueError("Ground truth shapes mismatch")
        if not np.allclose(mixtures, truth_mixtures, atol=1e-6):
            raise ValueError("Mixture data differs from original saved ground truth")
        if bits.shape[0] != mixtures.shape[0]:
            raise ValueError("Bit count/batch dimension mismatch")
        if not np.isfinite(mixtures).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite IQ found")

        net = Wave(cfg)
        # Official MIT checkpoints contain OmegaConf objects. Only use trusted files.
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["model"], strict=True)
        del ckpt
        net.to(device).eval()
        estimates = []
        with torch.inference_mode():
            for left in range(0, len(mixtures), args.batch_size):
                iq = mixtures[left:left + args.batch_size]
                x = np.stack([iq.real, iq.imag], axis=1).astype(np.float32)
                pred = net(torch.from_numpy(x).to(device))
                if pred.shape != (len(iq), 2, SAMPLE_SIZE):
                    raise RuntimeError("Wrong network output shape {}".format(tuple(pred.shape)))
                out_np = pred.detach().cpu().numpy()
                estimates.append((out_np[:, 0, :] + 1j * out_np[:, 1, :]).astype(np.complex64))
        estimate = np.concatenate(estimates, axis=0)
        if not np.isfinite(estimate).all():
            raise RuntimeError("WaveNet produced NaN/Inf")
        input_stats = stats(target, mixtures)
        output_stats = stats(target, estimate)
        sinrs = (10*np.log10(
            np.mean(np.abs(target.astype(np.complex128))**2, axis=1) /
            np.maximum(np.mean(np.abs((mixtures - target).astype(np.complex128))**2, axis=1), 1e-30)
        )).tolist()
        # Exact output filenames are used by subsequent official Sionna demod evaluation.
        pred_path = out / ("estimated_soi_{}_{}.npy".format(soi, INTERFERENCE))
        np.save(str(pred_path), estimate)
        rec = {
            "soi_type": soi,
            "interference_type": INTERFERENCE,
            "n_samples": int(len(mixtures)),
            "iq_shape": list(mixtures.shape),
            "bit_shape": list(bits.shape),
            "actual_input_sinr_db": sinrs,
            "input_waveform": input_stats,
            "wavenet_waveform": output_stats,
            "checkpoint": str(ckpt_path.relative_to(root)),
            "estimated_soi_file": str(pred_path.relative_to(root)),
        }
        summary["models"].append(rec)
        print("\n{} + {}: {} records".format(soi, INTERFERENCE, len(mixtures)), flush=True)
        for i in range(len(mixtures)):
            print("  record {} | SINR {:+.2f} dB | input MSE {:+.2f} dB | WaveNet MSE {:+.2f} dB | WaveNet NMSE {:+.2f} dB".format(
                i, sinrs[i], input_stats["mse_db"][i], output_stats["mse_db"][i], output_stats["nmse_db"][i]
            ), flush=True)
        print("Saved", pred_path, flush=True)
        del net
    summary_path = out / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print("\n=== SIONNA + WAVENET MINI INFERENCE COMPLETE ===", flush=True)
    print("Report:", summary_path, flush=True)
    print("Next: run official Sionna QPSK / OFDMQPSK demod and report BER", flush=True)


if __name__ == "__main__":
    main()
