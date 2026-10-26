#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe small-scale runner of MIT RF Challenge 0.2.0 ORIGINAL Sionna generator.

Place in signal-separation/, alongside rfchallenge/ and dataset/.

Uses official rfcutils and sampletest_testmixture_generator.py unchanged.
Only overrides the number of generated records, SINR grid, and output folder.
It NEVER overwrites original dataset, training assets or source files.

Command (inside a dedicated Sionna environment):
  CUDA_VISIBLE_DEVICES=-1 python mit_sionna_official_smoke.py

Default output: output/mit_sionna_official_smoke/dataset/*
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--soi-types', default='QPSK,OFDMQPSK', help='Comma-separated: QPSK,OFDMQPSK')
    p.add_argument('--interference', default='CommSignal2')
    p.add_argument('--samples-per-sinr', type=int, default=2)
    p.add_argument('--sinrs', default='-12', help='Comma-separated nominal SINRs in dB')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('output/mit_sionna_official_smoke'))
    args = p.parse_args()

    if not (1 <= args.samples_per_sinr <= 100):
        raise ValueError('Use 1 to 100 samples per SINR')
    soi_types = [v.strip().upper() for v in args.soi_types.split(',') if v.strip()]
    allowed_soi = {'QPSK', 'OFDMQPSK'}
    if not soi_types or any(v not in allowed_soi for v in soi_types) or len(set(soi_types)) != len(soi_types):
        raise ValueError('Only QPSK and OFDMQPSK are covered by published WaveNet checkpoints')
    sinrs = np.array([float(x.strip()) for x in args.sinrs.split(',')], dtype=np.float64)
    if not len(sinrs) or not np.isfinite(sinrs).all():
        raise ValueError('Invalid SINR list')
    if args.interference not in {'CommSignal2', 'CommSignal3', 'CommSignal5G1', 'EMISignal1'}:
        raise ValueError('Unknown interference type')

    root = args.root.resolve()
    official = root / 'rfchallenge'
    raw_frame = root / 'dataset' / 'testset1_frame' / f'{args.interference}_test1_raw_data.h5'
    gen_file = official / 'sampletest_testmixture_generator.py'
    for label, path in [('official generator', gen_file), ('interference HDF5', raw_frame)]:
        if not path.is_file():
            raise FileNotFoundError('{} missing: {}'.format(label, path))

    out = (args.out if args.out.is_absolute() else root / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty: {}. Select a fresh --out.'.format(out))
    workspace_data = out / 'dataset'
    workspace_data.mkdir(parents=True, exist_ok=True)
    link = workspace_data / 'testset1_frame'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(root / 'dataset' / 'testset1_frame', target_is_directory=True)
    if link.resolve() != (root / 'dataset' / 'testset1_frame').resolve():
        raise RuntimeError('testset1_frame path does not match the original dataset; refusing to continue')

    # Run TensorFlow on CPU to avoid old TF/CUDA conflicts on modern GPUs.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    import sionna as sn
    import h5py
    tf.config.set_visible_devices([], 'GPU')

    print('===== Official generator environment =====', flush=True)
    print('Python:', sys.version.split()[0])
    print('TensorFlow:', tf.__version__)
    print('Sionna:', getattr(sn, '__version__', 'version metadata unavailable'))
    print('NumPy:', np.__version__, 'h5py:', h5py.__version__)
    print('Sionna CPU execution:', tf.config.list_physical_devices('GPU') == [] or tf.config.get_visible_devices('GPU') == [])
    print('Generator:', gen_file)
    print('Original interference HDF5:', raw_frame)
    print('Isolated output:', out)

    sys.path.insert(0, str(official))
    spec = importlib.util.spec_from_file_location('mit_official_generator_v02', str(gen_file))
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load official generator module')
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.n_per_batch = args.samples_per_sinr
    gen.all_sinr = sinrs
    gen.seed_number = args.seed

    report = {'source': 'Official MIT RF Challenge 0.2.0 Sionna generator (mini-run)',
              'not_official_full_testset_score': True,
              'tensorflow': tf.__version__,
              'sionna': getattr(sn, '__version__', None),
              'interference': args.interference,
              'sinrs_db': sinrs.tolist(),
              'samples_per_sinr': args.samples_per_sinr,
              'seed': args.seed,
              'results': []}
    old_cwd = Path.cwd()
    try:
        os.chdir(str(out))
        for soi_type in soi_types:
            print('\n===== Official Sionna generation: {} + {} ====='.format(soi_type, args.interference), flush=True)
            gen.generate_demod_testmixture(soi_type, args.interference)
            stem = 'TestSet1Example_testmixture_{}_{}'.format(soi_type, args.interference)
            mix_path = workspace_data / (stem + '.npy')
            meta_path = workspace_data / (stem + '_metadata.npy')
            truth_path = workspace_data / 'GroundTruth_TestSet1Example_Dataset_{}_{}.pkl'.format(soi_type, args.interference)
            for f in (mix_path, meta_path, truth_path):
                if not f.is_file():
                    raise RuntimeError('Expected official output missing: {}'.format(f))
            mix = np.load(str(mix_path), allow_pickle=False)
            with truth_path.open('rb') as fp:
                truth_mix, truth_soi, truth_bits = pickle.load(fp)
            if mix.shape != (len(sinrs) * args.samples_per_sinr, 40960):
                raise RuntimeError('Unexpected mixture shape: {}'.format(mix.shape))
            if truth_soi.shape != mix.shape or truth_mix.shape != mix.shape:
                raise RuntimeError('SOI/mixture shape mismatch')
            if not np.all(np.isfinite(mix)) or not np.all(np.isfinite(truth_soi)):
                raise RuntimeError('NaN/Inf detected')
            if not np.allclose(mix, truth_mix):
                raise RuntimeError('Saved mixture differs from ground truth package')
            interf = mix - truth_soi
            sinr_actual = (10 * np.log10(
                np.mean(np.abs(truth_soi)**2, axis=1) / np.maximum(np.mean(np.abs(interf)**2, axis=1), 1e-30)
            )).tolist()
            row = {'soi_type': soi_type,
                   'mixture_shape': list(mix.shape),
                   'mixture_dtype': str(mix.dtype),
                   'soi_shape': list(truth_soi.shape),
                   'bits_shape': list(np.asarray(truth_bits).shape),
                   'actual_sinr_db': [round(float(x), 4) for x in sinr_actual],
                   'mixture_file': str(mix_path.relative_to(root)),
                   'groundtruth_file': str(truth_path.relative_to(root)),
                   'metadata_file': str(meta_path.relative_to(root))}
            report['results'].append(row)
            print('Mixture:', row['mixture_shape'], row['mixture_dtype'])
            print('SOI:', row['soi_shape'], 'True bits:', row['bits_shape'])
            print('Actual SINR (dB):', row['actual_sinr_db'])
    finally:
        os.chdir(str(old_cwd))
    summary_file = out / 'summary.json'
    summary_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n===== OFFICIAL SIONNA MINI-GENERATION COMPLETE =====')
    print('Report:', summary_file)
    print('No files in the original dataset/ directory were changed.')


if __name__ == '__main__':
    main()
