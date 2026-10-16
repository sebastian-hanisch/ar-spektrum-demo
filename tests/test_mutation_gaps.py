"""Gezielte Tests für Stellen, die der Mutationstest als ungeprüft meldete (Gleichstände, Randfälle, unabhängige Referenzfassungen, Reihenfolge der Zeilen)."""

import math

import numpy as np
import pytest

import arm_constants as C
import arm_evaluation as ev
import arm_order as O


def test_count_extra_peaks_plateaus_and_the_level_boundary():
    assert ev._count_extra_peaks([0.0, 1.0, 1.0, 0.0, 0.5, 0.0], -10.0) == 1        # Plateau: ein Gipfel (der letzte Punkt); dazu der Gipfel 0,5
    assert ev._count_extra_peaks([0.0, 1.0, 1.0, 0.0], -10.0) == 0                   # ein einziges Plateau ist ein Gipfel
    assert ev._count_extra_peaks([0.0, 1.0, 0.0, 0.1, 0.0], -10.0) == 0             # genau auf der Schwelle: zählt nicht
    assert ev._count_extra_peaks([0.0, 1.0, 0.0, 0.1001, 0.0], -10.0) == 1


def test_rmse_uses_the_window_between_four_pad_points_from_each_end():
    pad = 2
    true = np.ones(40)
    est = np.ones(40)
    est[7] = 100.0                                                                    # liegt vor dem Fenster: zählt nicht
    est[32] = 100.0                                                                   # liegt hinter dem Fenster (Indizes 8 bis 31): zählt nicht
    assert ev._rmse(est, true, pad) == 0.0
    est[8] = 2.0
    assert ev._rmse(est, true, pad) == pytest.approx(math.sqrt((10.0 / math.log(10.0) * math.log(2.0)) ** 2 / 24.0), abs=1e-12)
    est = np.ones(40)
    est[31] = 2.0
    assert ev._rmse(est, true, pad) == pytest.approx(math.sqrt((10.0 / math.log(10.0) * math.log(2.0)) ** 2 / 24.0), abs=1e-12)


def test_analysis_welch_segment_length_and_hann_periodogram_by_a_direct_formula():
    for n, seg in ((64, 16), (128, 32), (1024, 256)):
        assert ev.analyse(ev.Settings(n=n)).welch_seg == seg
    a = ev.analyse(ev.Settings(n=128))
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(128) / 128)
    ref = np.abs(np.fft.rfft(a.x * w, C.PAD * 128)) ** 2 / (C.FS * (w ** 2).sum())
    assert a.fft == pytest.approx(ref, rel=1e-10)


def test_analysis_estimated_poles_belong_to_the_fitted_order():
    a = ev.analyse(ev.Settings(order_mode="fest", order=5))
    assert len(a.est_poles) == 5 and a.max_radius == pytest.approx(float(np.abs(a.est_poles).max()))


def test_verdict_rows_are_selected_by_both_conditions():
    rows = [{"n": 128, "background": "Resonanz", "welch": 9.0, "multitaper": 9.0, "ar_true": 9.0, "yw_true": 9.0},
            {"n": 128, "background": "Steil", "welch": 3.25, "multitaper": 4.92, "ar_true": 1.02, "yw_true": 10.23},
            {"n": 1024, "background": "Resonanz", "welch": 9.0, "multitaper": 9.0, "ar_true": 9.0, "yw_true": 9.0},
            {"n": 1024, "background": "Steil", "welch": 1.0, "multitaper": 1.0, "ar_true": 0.4, "yw_true": 4.65}]
    assert "mit Welch 3,25 dB" in ev.verdict_best(rows)[1] and "(n = 1024: 4,7 dB)" in ev.verdict_best(rows)[1]
    split = [{"snr": 20.0, "method": "Burg", "orders": [2, 4], "shares": [0.9, 0.9]}, {"snr": 60.0, "method": "Yule-Walker", "orders": [2, 4], "shares": [0.9, 0.9]},
             {"snr": 60.0, "method": "Burg", "orders": [2, 4], "shares": [0.0, 0.18]}, {"snr": 20.0, "method": "Kovarianz", "orders": [4, 12], "shares": [0.9, 0.9]},
             {"snr": 60.0, "method": "Kovarianz", "orders": [4, 12], "shares": [0.0, 0.31]}]
    assert "Ordnung 4 zeigt Burg in 18 %" in ev.verdict_splitting(split)[1] and "Ordnung 12 in 31 %" in ev.verdict_splitting(split)[1]


def test_select_order_default_orders_start_at_one():
    assert O.select_order([3.0, 5.0, 4.0]) == 1 and O.select_order([5.0, 4.0, 3.0]) == 3 and O.select_order([5.0, 3.0, 4.0]) == 2
