#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the FINAL local Sionna WaveNet BER evaluation by nominal SINR.

Pure standard-library; compatible with Python 3.7 and newer.
Run from signal-separation/:
  python summarize_sinr.py --generated output/mit_sionna_seed42 \
      --evaluated output/mit_ber_seed42 --out output/mit_report_seed42

Input: generator summary.json and official_ber per_sample.csv.
Output: one CSV row per (SOI type, nominal SINR); corresponding summary.json.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--generated', type=Path, required=True)
    ap.add_argument('--evaluated', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    generated, evaluated, out = (x.resolve() for x in (args.generated, args.evaluated, args.out))
    gen = json.loads((generated / 'summary.json').read_text(encoding='utf-8'))
    sinrs = gen['sinrs_db']
    per_sinr = int(gen['samples_per_sinr'])
    soi_types = sorted(x['soi_type'] for x in gen['results'])
    if len(set(soi_types)) != len(soi_types) or per_sinr <= 0 or not sinrs:
        raise ValueError('Invalid generator summary: SOI/sinr/sample-count')
    with (evaluated / 'per_sample.csv').open('r', newline='', encoding='utf-8') as fp:
        rows = list(csv.DictReader(fp))
    expected = len(soi_types) * len(sinrs) * per_sinr
    if len(rows) != expected:
        raise ValueError('Expected {} rows, got {}'.format(expected, len(rows)))
    groups = defaultdict(list)
    for r in rows:
        soi = r['soi_type']
        sample_index = int(r['sample_index'])
        bit_count = int(r['bit_count'])
        err_in = int(r['input_bit_errors'])
        err_out = int(r['wavenet_bit_errors'])
        if soi not in soi_types or sample_index < 0 or sample_index >= len(sinrs) * per_sinr:
            raise ValueError('Unexpected soi_type or sample_index in BER output')
        if not (0 <= err_in <= bit_count and 0 <= err_out <= bit_count):
            raise ValueError('Invalid error-count or bit-count')
        groups[(soi, sample_index // per_sinr)].append(r)

    output = []
    for soi in soi_types:
        for idx, nominal_sinr in enumerate(sinrs):
            group = groups[(soi, idx)]
            actual_indices = sorted(int(r['sample_index']) for r in group)
            expected_indices = list(range(idx*per_sinr, (idx+1)*per_sinr))
            if actual_indices != expected_indices:
                raise ValueError('Missing/duplicate indices for {} SINR {}'.format(soi, nominal_sinr))
            total_bits = sum(int(r['bit_count']) for r in group)
            in_errs = sum(int(r['input_bit_errors']) for r in group)
            out_errs = sum(int(r['wavenet_bit_errors']) for r in group)
            in_mse = [10**(float(r['input_mse_db'])/10) for r in group]
            out_mse = [10**(float(r['wavenet_mse_db'])/10) for r in group]
            if not all(math.isfinite(x) and x>=0 for x in in_mse+out_mse):
                raise ValueError('Invalid per-sample MSE values')
            db = lambda x: 10*math.log10(max(x,1e-30))
            output.append({
                'soi_type':soi,
                'interference':gen['interference'],
                'nominal_sinr_db':float(nominal_sinr),
                'samples':len(group),
                'mean_actual_sinr_db':sum(float(r['actual_input_sinr_db']) for r in group)/len(group),
                'bits_evaluated':total_bits,
                'input_bit_errors':in_errs,
                'wavenet_bit_errors':out_errs,
                'input_ber':in_errs/total_bits,
                'wavenet_ber':out_errs/total_bits,
                'worst_wavenet_ber':max(float(r['wavenet_ber']) for r in group),
                'zero_bit_error_samples':sum(int(r['wavenet_bit_errors']) == 0 for r in group),
                'mean_input_mse_db':db(sum(in_mse)/len(in_mse)),
                'mean_wavenet_mse_db':db(sum(out_mse)/len(out_mse)),
            })
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Report directory not empty: {}. Choose a new --out.'.format(out))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'by_sinr.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)
    report={
        'description':'Local MIT RF Challenge 0.2.0 Sionna-generated WaveNet benchmark (not official hidden benchmark)',
        'provenance':{'generated_manifest':str((generated/'summary.json').resolve()),'per_sample_csv':str((evaluated/'per_sample.csv').resolve()),'seed':gen['seed']},
        'formulas':{
            'ber':'sum(bit_errors) / sum(bit_count)',
            'mean_mse_db':'10 * log10(mean(linear per-sample MSE))',
            'zero_ber_note':'zero observed bit errors, not a theoretical guarantee'
        },
        'rows':output,
    }
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved {} SINR summaries to {}'.format(len(output),out))
    for x in output:
        print('{:9} SINR {:>5g} dB: input BER {:7.3%}, recovered {:7.3%}, MSE {:7.2f} dB'.format(
            x['soi_type'],x['nominal_sinr_db'],x['input_ber'],x['wavenet_ber'],x['mean_wavenet_mse_db']))


if __name__=='__main__':
    main()
