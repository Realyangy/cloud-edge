#!/usr/bin/env python3
"""Paired YOLO11n -> XCiT benchmark (TorchSig 2.1.1, 57 classes).

Tests the SAME YOLO predicted boxes and SAME 4096-IQ extraction with and without
complex RMS normalization. Runs both models only once, and creates exactly one
sequential seeded TorchSig wideband validation-style stream.

Prerequisite: combined_inference.py has the corrected NEGATIVE frequency shift
and the normalization changes, including quality['rms_before_normalization'].

Usage:
  CUDA_VISIBLE_DEVICES=1 python benchmark_combined.py --samples 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchsig
from ultralytics import YOLO
from torchsig_models.models import XCiTClassifier
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.yaml import load_config_from_yaml

from inference import as_signal_list, box_features, evaluate, to_yolo_image, xywh_to_xyxy
from combined_inference import CLASS_NAMES, extract_iq, classify_iq, signal_family

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.5, help='IoU used to match detections with GT')
    p.add_argument('--config', type=Path, default=ROOT/'configs/wideband_clean_val_all.yaml')
    p.add_argument('--yolo-weights', type=Path, default=ROOT/'models/yolo11n_wideband_v1.0.0.pt')
    p.add_argument('--xcit-weights', type=Path,
                   default=ROOT/'models/finetuned/xcit_joint_v3.ckpt')
    p.add_argument('--device', default='auto', help='auto, cpu, 0 after CUDA_VISIBLE_DEVICES mapping')
    p.add_argument('--rms-targets', type=float, nargs='+', default=[1.0],
                   help='Complex RMS values to test after normalization (default: 1.0)')
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args()


def make_stream(config_path: Path, seed: int):
    cfg = load_config_from_yaml(config_path)
    if (cfg.dataset_id, cfg.output_representation, cfg.impairment_level,
        cfg.signal_sampling_mode) != ('wideband_clean_val_all', 'spectrogram', 0, 'per_signal'):
        raise ValueError('Expected official TorchSig 2.1.1 wideband_clean_val_all settings')
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(cfg.dataset_metadata)
    iq_capture = {}

    def capture_iq(signal):
        iq = np.asarray(signal.data)
        if not np.iscomplexobj(iq):
            raise RuntimeError('Expected complex IQ before Spectrogram transform')
        iq_capture['raw'] = np.asarray(iq, dtype=np.complex64).copy()
        return signal

    imp = Impairments(level=cfg.impairment_level)
    ds = TorchSigIterableDataset(
        signal_generators='all', metadata=metadata,
        transforms=[imp.dataset_transforms, capture_iq,
                    Spectrogram(fft_size=int(metadata['fft_size'])), YOLOLabel()],
        component_transforms=[imp.signal_transforms],
        target_labels=['yolo_label', 'class_name'],
    )
    loader = WorkerSeedingDataLoader(ds, batch_size=1, num_workers=0,
                                     collate_fn=lambda batch: batch, seed=seed)
    loader.seed(seed)
    return iter(loader), iq_capture, metadata, cfg


def per_mode_metrics(totals: dict, gt_family_counts: Counter) -> dict:
    considered = int(totals['matched_classified'])
    all_gt = int(totals['gt'])
    return {
        'matched_classified': considered,
        'family_correct': int(totals['family_correct']),
        'fine_correct': int(totals['fine_correct']),
        'family_accuracy_on_matched': totals['family_correct']/considered if considered else None,
        'fine_accuracy_on_matched': totals['fine_correct']/considered if considered else None,
        'family_end_to_end_recall_on_all_gt': totals['family_correct']/all_gt if all_gt else None,
        'fine_end_to_end_recall_on_all_gt': totals['fine_correct']/all_gt if all_gt else None,
        'family_by_gt': {
            fam: {
                'gt_count': int(gt_family_counts[fam]),
                'matched_classified': int(totals['by_family'][fam]['matched']),
                'family_correct': int(totals['by_family'][fam]['family_correct']),
                'fine_correct': int(totals['by_family'][fam]['fine_correct']),
            } for fam in sorted(gt_family_counts)
        }
    }


def main():
    args = parse_args()
    if torchsig.__version__ != '2.1.1':
        raise RuntimeError(f'Expected torchsig==2.1.1, got {torchsig.__version__}')
    if args.samples < 1 or not (0 < args.conf <= 1) or not (0 < args.iou <= 1):
        raise ValueError('--samples must be >=1 and --conf/--iou must be within (0,1]')
    if not args.rms_targets or any(not np.isfinite(x) or x <= 0 for x in args.rms_targets):
        raise ValueError('--rms-targets must contain positive finite values')
    if len(set(args.rms_targets)) != len(args.rms_targets):
        raise ValueError('--rms-targets must not contain duplicates')
    cfg = load_config_from_yaml(args.config)
    seed = int(cfg.seed if args.seed is None else args.seed)
    out = (args.out or ROOT/'output'/f'combined_batch{args.samples}_paired_seed{seed}').resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing results: {out}')
    out.mkdir(parents=True, exist_ok=True)

    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch_device = 'cuda:0' if device == '0' else (f'cuda:{device}' if device.isdigit() else device)
    yolo = YOLO(str(args.yolo_weights))
    xcit = XCiTClassifier.load_from_checkpoint(str(args.xcit_weights), map_location='cpu',
                                                weights_only=False).to(torch_device).eval()
    if int(xcit.hparams.num_classes) != len(CLASS_NAMES) or int(xcit.hparams.input_channels) != 2:
        raise RuntimeError('XCiT checkpoint class/channel count mismatch')

    stream, capture, md, _ = make_stream(args.config, seed)
    fs = float(md['sample_rate'])
    fft_size = int(md['fft_size'])
    duration_s = int(md['num_iq_samples_dataset']) / fs
    total = {'gt':0,'pred':0,'tp':0,'fp':0,'fn':0}
    gt_family_counts = Counter()
    rms_modes = [('with_rms' if value == 1.0 else f'rms_target_{value:g}', value)
                 for value in args.rms_targets]
    modes = ('without_rms', *(name for name, _ in rms_modes))
    stats = {
        mode: {'gt':0,'matched_classified':0,'family_correct':0,'fine_correct':0,
               'by_family':defaultdict(lambda:{'matched':0,'family_correct':0,'fine_correct':0})}
        for mode in modes
    }
    records = []
    errors = []

    for index in range(args.samples):
        capture.clear()
        spec, target = next(stream)[0]
        if 'raw' not in capture:
            raise RuntimeError(f'No IQ was captured for sample {index}')
        raw = capture['raw']
        bgr = to_yolo_image(np.asarray(spec, dtype=np.float32))
        if bgr.shape != (fft_size, fft_size, 3):
            raise RuntimeError(f'Unexpected image shape {bgr.shape}')
        gt_boxes = as_signal_list(target[0], 'box')
        gt_names = [str(n) for n in as_signal_list(target[1], 'name')]
        if len(gt_boxes) != len(gt_names):
            raise RuntimeError(f'Mismatched GT count in sample {index}')
        gt = []
        for label, name in zip(gt_boxes,gt_names):
            _, cx, cy, bw, bh = map(float,label)
            gt.append({'true_class':name, 'box_xyxy_normalized':xywh_to_xyxy(cx,cy,bw,bh),
                       'features': box_features(xywh_to_xyxy(cx,cy,bw,bh),fs,duration_s)})
            gt_family_counts[signal_family(name)] += 1

        res = yolo.predict(source=bgr, imgsz=fft_size, conf=args.conf,
                           device=device, verbose=False)[0]
        preds = []
        h,w = bgr.shape[:2]
        for j, (b, score) in enumerate(zip(res.boxes.xyxy.cpu().numpy(),
                                            res.boxes.conf.cpu().numpy())):
            a,bb,c,d = map(float,b)
            box = [a/w,bb/h,c/w,d/h]
            rec = {'prediction_index':j,'confidence':float(score),
                   'box_xyxy_normalized':box}
            try:
                x_norm, quality = extract_iq(raw,box,fs)
                if quality.get('normalization') != 'complex_rms' or 'rms_before_normalization' not in quality:
                    raise RuntimeError('Current combined_inference.py must use complex RMS normalization')
                rms = float(quality['rms_before_normalization'])
                if not np.isfinite(rms) or rms < 1e-8:
                    raise ValueError(f'Unexpected pre-normalization RMS: {rms}')
                x_raw = np.asarray(x_norm*rms,dtype=np.float32)
                # Every mode shares exactly the same detection and channelization.
                rec['without_rms'] = classify_iq(xcit,x_raw,torch_device)
                for mode_name, target_rms in rms_modes:
                    rec[mode_name] = classify_iq(
                        xcit, np.asarray(x_norm * target_rms, dtype=np.float32), torch_device
                    )
                rec['pre_normalization_rms'] = rms
                rec['extraction'] = quality
            except (ValueError,RuntimeError) as exc:
                rec['error'] = str(exc)
                errors.append({'sample_index':index,'prediction_index':j,'error':str(exc)})
                if 'must use complex RMS' in str(exc):
                    raise
            preds.append(rec)
        matched = evaluate(preds,gt,args.iou)
        total['gt'] += len(gt)
        total['pred'] += len(preds)
        for name,key in [('TP','tp'),('FP','fp'),('FN','fn')]:
            total[key] += int(matched[name])
        for mode in modes:
            stats[mode]['gt'] += len(gt)
        for pair in matched['matches']:
            pidx,gidx = pair['prediction_index'],pair['ground_truth_index']
            actual = gt[gidx]['true_class']
            actual_fam = signal_family(actual)
            for mode in modes:
                result = preds[pidx].get(mode)
                if result is None:
                    continue
                cell = stats[mode]
                cell['matched_classified'] += 1
                cell['by_family'][actual_fam]['matched'] += 1
                fam_ok = result['family_top1'] == actual_fam
                fine_ok = result['fine_top1'] == actual
                cell['family_correct'] += int(fam_ok)
                cell['fine_correct'] += int(fine_ok)
                cell['by_family'][actual_fam]['family_correct'] += int(fam_ok)
                cell['by_family'][actual_fam]['fine_correct'] += int(fine_ok)
                preds[pidx].setdefault('ground_truth',{})['class'] = actual
                preds[pidx]['ground_truth']['family'] = actual_fam
        records.append({'sample_index':index,'ground_truth_count':len(gt),
                        'prediction_count':len(preds),'TP':int(matched['TP']),
                        'FP':int(matched['FP']),'FN':int(matched['FN']),
                        'predictions':preds})
        if (index+1) % 10 == 0 or index+1 == args.samples:
            score_text = ' '.join(
                f'{mode}={stats[mode]["family_correct"]}/{stats[mode]["matched_classified"]}'
                for mode in modes
            )
            print(f'Processed {index+1}/{args.samples} | GT={total["gt"]} '
                  f'TP={total["tp"]} | matched families: {score_text}', flush=True)

    precision = total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else None
    recall = total['tp']/(total['tp']+total['fn']) if total['tp']+total['fn'] else None
    summary = {
        'dataset':'TorchSig 2.1.1 official clean validation CONFIG, freshly generated; not official stored holdout',
        'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'seed':seed,'samples':args.samples,'confidence_threshold':args.conf,
        'rms_targets':args.rms_targets,
        'gt_matching_iou':args.iou,'models':{'yolo':args.yolo_weights.name,'xcit':args.xcit_weights.name},
        'detection':{**total,'precision_at_fixed_conf':precision,'recall_at_fixed_conf':recall},
        'classification':{mode:per_mode_metrics(stats[mode],gt_family_counts) for mode in modes},
        'extraction_errors':len(errors),
        'notes':['All classifications are evaluated on identical YOLO predicted boxes.',
                 'Unnormalized input is reconstructed by multiplying the RMS-normalized IQ by its original RMS.',
                 'Each rms_target mode scales the same unit-RMS extracted IQ to the requested complex RMS.',
                 'Family is computed by SUMMING the 57 output-class probabilities, not mapping fine Top-1.',
                 'mAP is NOT reported here; the existing benchmark_yolo.py computes detection mAP.',
                 'Matched classification accuracy and end-to-end recall have different denominators.'],
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'extraction_errors.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n=== Paired comparison on identical predicted boxes ===')
    print(f'Samples: {args.samples}; GT={total["gt"]}; Pred={total["pred"]}; TP={total["tp"]}, FP={total["fp"]}, FN={total["fn"]}')
    print(f'YOLO Precision@conf={args.conf}: {precision:.2%}' if precision is not None else 'YOLO Precision: N/A')
    print(f'YOLO Recall@conf={args.conf}: {recall:.2%}' if recall is not None else 'YOLO Recall: N/A')
    for mode in modes:
        t=summary['classification'][mode]
        print(f'{mode}: matched family {t["family_correct"]}/{t["matched_classified"]} '
              f'= {t["family_accuracy_on_matched"]:.2%}' if t['family_accuracy_on_matched'] is not None
              else f'{mode}: no matched classified instances')
        print(f'         matched fine {t["fine_correct"]}/{t["matched_classified"]}; '
              f'end-to-end family recall {t["family_correct"]}/{total["gt"]}')
    print(f'Extraction failures: {len(errors)}')
    print('Saved:',out)


if __name__=='__main__':
    main()
