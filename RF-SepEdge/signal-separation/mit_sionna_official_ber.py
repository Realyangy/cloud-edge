#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MIT RF Challenge 0.2.0 Sionna mini-set: original demodulation and BER.

Run in the SEPARATE Python 3.7 rf-sionna environment, CPU only:
    CUDA_VISIBLE_DEVICES=-1 python mit_sionna_official_ber.py

Before running, generate a two-sample Sionna mini-set with
mit_sionna_official_smoke.py and infer with mit_sionna_wavenet_infer.py.

Uses UNMODIFIED official rfcutils.qpsk_matched_filter_demod and
rfcutils.ofdm_demod, just as in sampletest_torch_wavenet_inference.py.
MSE and BER formulas match official sampletest_evaluationscript.py,
but this is a 2-sample smoke test, NOT the full 1100-record challenge score.

Only reads locally generated trusted ground truth .pkl; never open untrusted
pickle files. Does not edit official source, models, or data.
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

# Keep TensorFlow CPU-only before importing it.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

SIGNAL_LENGTH = 40960
SOI_TYPES = ('QPSK', 'OFDMQPSK')
INTERFERENCE = 'CommSignal2'
BITS_PER_SIGNAL = {'QPSK': 5120, 'OFDMQPSK': 57344}


def mean_power(z):
    return np.mean(np.abs(np.asarray(z, dtype=np.complex128)) ** 2, axis=1)


def per_record_mse(reference, estimate):
    return mean_power(np.asarray(estimate, dtype=np.complex128) -
                      np.asarray(reference, dtype=np.complex128))


def power_to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def as_binary_bits(x, expected_shape, name):
    a = np.asarray(x)
    if a.shape != expected_shape:
        raise ValueError('{} shape {} != {}'.format(name, a.shape, expected_shape))
    if not np.isfinite(a).all():
        raise ValueError('{} contains NaN/Inf'.format(name))
    if not np.all(np.logical_or(a == 0, a == 1)):
        raise ValueError('{} contains values other than 0 and 1'.format(name))
    return a.astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--source', type=Path, default=Path('output/mit_sionna_official_smoke'))
    p.add_argument('--predictions', type=Path, default=Path('output/mit_sionna_wavenet_infer'))
    p.add_argument('--out', type=Path, default=Path('output/mit_sionna_official_ber'))
    args = p.parse_args()
    root = args.root.resolve()
    source = (args.source if args.source.is_absolute() else root / args.source).resolve()
    predicted = (args.predictions if args.predictions.is_absolute() else root / args.predictions).resolve()
    out = (args.out if args.out.is_absolute() else root / args.out).resolve()
    official = root / 'rfchallenge'
    manifest_path = source / 'summary.json'
    inference_summary_path = predicted / 'summary.json'
    for f in [manifest_path, inference_summary_path, official / 'rfcutils/__init__.py']:
        if not f.is_file():
            raise FileNotFoundError('Missing: {}'.format(f))
    with manifest_path.open('r', encoding='utf-8') as f:
        manifest = json.load(f)
    with inference_summary_path.open('r', encoding='utf-8') as f:
        infer_summary = json.load(f)
    if manifest.get('interference') != INTERFERENCE:
        raise ValueError('Expected {} generated mini-set'.format(INTERFERENCE))
    if set(v['soi_type'] for v in manifest.get('results', [])) != set(SOI_TYPES):
        raise ValueError('Missing QPSK or OFDMQPSK in generator summary')
    if set(v['soi_type'] for v in infer_summary.get('models', [])) != set(SOI_TYPES):
        raise ValueError('Missing QPSK or OFDMQPSK in inference summary')

    import tensorflow as tf
    import sionna
    tf.config.set_visible_devices([], 'GPU')
    sys.path.insert(0, str(official))
    import rfcutils

    print('===== 1. Sionna official demodulation check =====', flush=True)
    print('Python:', sys.version.split()[0], 'TF:', tf.__version__,
          'Sionna:', getattr(sionna, '__version__', 'unknown'))
    print('CPU-only:', not bool(tf.config.get_visible_devices('GPU')))
    print('Generated source:', source)
    print('WaveNet predictions:', predicted)
    print('BER: direct comparison with official generated source bits', flush=True)

    # Same OFDM resource grid used by the official WaveNet inference script.
    # The helper constructs a grid for 40960/(64 FFT + 16 CP) = 512 symbols.
    resource_grid = rfcutils.get_resource_grid(SIGNAL_LENGTH // 80)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty: {}. Select a fresh --out.'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    results = []
    for soi in SOI_TYPES:
        stem = 'TestSet1Example_testmixture_{}_{}'.format(soi, INTERFERENCE)
        mix_file = source / 'dataset' / (stem + '.npy')
        truth_file = source / 'dataset' / 'GroundTruth_TestSet1Example_Dataset_{}_{}.pkl'.format(soi, INTERFERENCE)
        pred_file = predicted / ('estimated_soi_{}_{}.npy'.format(soi, INTERFERENCE))
        for f in (mix_file, truth_file, pred_file):
            if not f.is_file():
                raise FileNotFoundError(f)
        mix = np.load(str(mix_file), allow_pickle=False)
        pred = np.load(str(pred_file), allow_pickle=False)
        # Only locally created pickle from previously successful Sionna mini-run.
        with truth_file.open('rb') as f:
            truth_mix, target, true_bits = pickle.load(f)
        mix = np.asarray(mix, dtype=np.complex64)
        pred = np.asarray(pred, dtype=np.complex64)
        target = np.asarray(target, dtype=np.complex64)
        if mix.ndim != 2 or mix.shape[1] != SIGNAL_LENGTH:
            raise ValueError('Unexpected mixed IQ shape {}'.format(mix.shape))
        if target.shape != mix.shape or pred.shape != mix.shape:
            raise ValueError('Mixture/reference/prediction shape mismatch')
        if not np.allclose(np.asarray(truth_mix), mix):
            raise ValueError('Input mixture differs from saved ground truth')
        if not np.isfinite(mix).all() or not np.isfinite(pred).all():
            raise ValueError('Nonfinite IQ detected')
        n = len(mix)
        bit_shape = (n, BITS_PER_SIGNAL[soi])
        true_bits = as_binary_bits(true_bits, bit_shape, soi + ' ground truth bits')

        def demod(arr):
            x = tf.convert_to_tensor(arr, dtype=tf.complex64)
            if soi == 'QPSK':
                estimated_bits, _ = rfcutils.qpsk_matched_filter_demod(x)
            else:
                estimated_bits, _ = rfcutils.ofdm_demod(x, resource_grid)
            return as_binary_bits(estimated_bits.numpy(), bit_shape,
                                  soi + ' demodulated bits')

        # Sanity check: the ORIGINAL Sionna-generated target should demodulate
        # to its own stored bits, otherwise do not trust BER comparisons.
        truth_decoded = demod(target)
        truth_errors = np.count_nonzero(truth_decoded != true_bits, axis=1)
        if np.any(truth_errors):
            raise RuntimeError('{} clean SOI has {} bit errors. Check official '
                               'grid/demod/ground truth before comparing model.'
                               .format(soi, truth_errors.tolist()))

        mix_bits = demod(mix)
        pred_bits = demod(pred)
        # Same per-record MSE/BER formulas as official sampletest_evaluationscript.py.
        mse_input = per_record_mse(target, mix)
        mse_pred = per_record_mse(target, pred)
        reference_power = mean_power(target)
        actual_sinr_db = power_to_db(reference_power / np.maximum(mse_input, 1e-30))
        ber_input = np.mean(mix_bits != true_bits, axis=1)
        ber_pred = np.mean(pred_bits != true_bits, axis=1)
        nmse_pred_db = power_to_db(mse_pred / np.maximum(reference_power, 1e-30))
        print('\n===== {} + {}: {} samples ====='.format(soi, INTERFERENCE, n), flush=True)
        for i in range(n):
            row = {
                'soi_type': soi, 'interference': INTERFERENCE,
                'sample_index': i, 'bit_count': BITS_PER_SIGNAL[soi],
                'actual_input_sinr_db': float(actual_sinr_db[i]),
                'input_mse_db': float(power_to_db(mse_input[i])),
                'wavenet_mse_db': float(power_to_db(mse_pred[i])),
                'wavenet_nmse_db': float(nmse_pred_db[i]),
                'input_bit_errors': int(np.count_nonzero(mix_bits[i] != true_bits[i])),
                'input_ber': float(ber_input[i]),
                'wavenet_bit_errors': int(np.count_nonzero(pred_bits[i] != true_bits[i])),
                'wavenet_ber': float(ber_pred[i]),
            }
            all_rows.append(row)
            print('record {} | SINR {:+.2f} dB | Input BER {:8.3%} | WaveNet BER {:8.3%} | WaveNet MSE {:+7.2f} dB'.format(
                i, row['actual_input_sinr_db'], row['input_ber'],
                row['wavenet_ber'], row['wavenet_mse_db']), flush=True)

        bits_file = out / 'estimated_bits_{}_{}.npy'.format(soi, INTERFERENCE)
        np.save(str(bits_file), pred_bits)
        results.append({
            'soi_type': soi, 'interference': INTERFERENCE,
            'samples': n, 'bit_count_per_sample': BITS_PER_SIGNAL[soi],
            'mean_input_ber': float(np.mean(ber_input)),
            'mean_wavenet_ber': float(np.mean(ber_pred)),
            # Official script takes dB AFTER averaging linear MSE.
            'mean_wavenet_mse_db_official_style': float(power_to_db(np.mean(mse_pred))),
            'mean_input_mse_db_official_style': float(power_to_db(np.mean(mse_input))),
            'clean_target_reference_errors': 0,
            'prediction_bits_file': str(bits_file.relative_to(root)),
        })
        print('Mean input BER {:8.3%} | Mean WaveNet BER {:8.3%} | Mean WaveNet MSE {:+.2f} dB'.format(
            results[-1]['mean_input_ber'], results[-1]['mean_wavenet_ber'],
            results[-1]['mean_wavenet_mse_db_official_style']), flush=True)
    with (out / 'per_sample.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        'description': 'MIT RF Challenge 0.2.0 Sionna demod helpers + MSE/BER definitions on {} locally generated records'.format(len(all_rows)),
        'official_full_benchmark': False,
        'source': str(source),
        'predictions': str(predicted),
        'demod_helpers': ['rfcutils.qpsk_matched_filter_demod',
                          'rfcutils.ofdm_demod'],
        'notes': ['Clean SOI reference must decode with zero errors',
                  'Sionna-generated targets and official weights',
                  'Sample size too small for an official benchmark performance claim'],
        'results': results,
    }
    with (out / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('\n===== OFFICIAL SIONNA BER MINI-EVALUATION COMPLETE =====', flush=True)
    print('Outputs:', out)
    print(' - summary.json / per_sample.csv / estimated_bits_*.npy')


if __name__ == '__main__':
    main()
