#!/usr/bin/env python3
"""Physical regression tests for YOLO-box IQ downconversion.

These tests use deterministic complex tones and do not load either model.
They verify the spectrogram-box frequency convention for signals on both sides
of DC, so a future frequency-shift sign regression fails immediately.
"""
from __future__ import annotations

import unittest

import numpy as np

from combined_inference import extract_iq


class ExtractIqFrequencyShiftTest(unittest.TestCase):
    SAMPLE_RATE = 10_000_000.0
    NUM_SAMPLES = 65_536
    BOX_BANDWIDTH_HZ = 200_000.0

    def _assert_tone_moves_to_dc(self, tone_hz: float) -> None:
        time_s = np.arange(self.NUM_SAMPLES, dtype=np.float64) / self.SAMPLE_RATE
        raw = np.exp(2j * np.pi * tone_hz * time_s).astype(np.complex64)

        # combined_inference.py uses center_hz = (0.5 - y_center) * fs.
        y_center = 0.5 - tone_hz / self.SAMPLE_RATE
        half_height = 0.5 * self.BOX_BANDWIDTH_HZ / self.SAMPLE_RATE
        box = [0.0, y_center - half_height, 1.0, y_center + half_height]

        channels, quality = extract_iq(raw, box, self.SAMPLE_RATE)
        iq = channels[0].astype(np.float64) + 1j * channels[1].astype(np.float64)
        output_rate = float(quality["estimated_output_rate_hz"])
        spectrum = np.fft.fftshift(np.fft.fft(iq * np.hanning(len(iq))))
        frequencies = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1.0 / output_rate))
        peak_hz = float(frequencies[int(np.argmax(np.abs(spectrum)))])
        one_bin_hz = output_rate / len(iq)

        self.assertLessEqual(abs(peak_hz), 1.5 * one_bin_hz)
        self.assertAlmostEqual(
            float(quality["predicted_center_offset_hz"]), tone_hz, delta=1.0
        )
        self.assertGreater(float(quality["rms_before_normalization"]), 0.90)

    def test_positive_frequency_tone_moves_to_dc(self) -> None:
        self._assert_tone_moves_to_dc(1_700_000.0)

    def test_negative_frequency_tone_moves_to_dc(self) -> None:
        self._assert_tone_moves_to_dc(-1_700_000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
