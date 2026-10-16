"""Vergleichsverfahren (wortgleich aus den Vorgänger-Demos): Segmentierung, Welch, Hann-Fenster, Slepian-Taper, Multitaper, Auflösungskriterium, Stellen, die der Mutationstest meldete."""

import numpy as np
import pytest
from scipy.signal import windows as spw

import arm_compare as H


def test_segment_counts_and_hops():
    assert H.n_segments(8, 4, 2) == 3 and H.n_segments(8, 4, 4) == 2 and H.n_segments(4, 4, 1) == 1 and H.n_segments(7, 4, 2) == 2
    assert H.n_segments(10, 4, 1) == 7 and H.n_segments(10, 4, 3) == 3
    with pytest.raises(ValueError):
        H.n_segments(3, 4, 1)
    with pytest.raises(ValueError):
        H.n_segments(8, 4, 0)
    assert H.n_segments(8, 8, 1) == 1                                          # Segment gleich Folge ist erlaubt
    assert H.hop_for(256, 0.5) == 128 and H.hop_for(256, 0.0) == 256 and H.hop_for(32, 0.99) == 1


def test_welch_of_a_constant_by_hand():
    w = np.array([0.0, 0.5, 1.0, 0.5])
    est, k = H.welch(np.ones(8), 4, 2, w, 1.0)
    assert k == 3 and est == pytest.approx([4.0 / 1.5, 1.0 / 1.5, 0.0], abs=1e-12)
    with pytest.raises(ValueError):
        H.welch(np.ones(8), 8, 4, w, 1.0)


def test_windows_and_multitaper_by_a_direct_loop():
    assert H.window("Rechteck", 4).tolist() == [1.0] * 4 and H.window("Hann", 4) == pytest.approx([0.0, 0.5, 1.0, 0.5])
    with pytest.raises(ValueError):
        H.window("Gauß", 4)
    h = H.dpss(16, 2, 3)
    x = np.random.default_rng(4).normal(size=16)
    y = H.eigenspectra(x, h, 32)
    est = H.multitaper_equal(x, h, 2.0, 32)
    for k in range(3):
        for j in (0, 5, 16):
            assert y[k, j] == pytest.approx(sum(h[k, t] * x[t] * np.exp(-2j * np.pi * j * t / 32) for t in range(16)), abs=1e-12)
    assert est == pytest.approx((np.abs(y) ** 2).mean(axis=0) / 2.0, abs=1e-12)


@pytest.mark.parametrize("n,nw,k", [(3, 1, 3), (16, 2, 3), (64, 2.5, 4), (128, 4, 7), (257, 3, 5), (512, 6, 11)])
def test_dpss_equals_scipy(n, nw, k):
    assert H.dpss(n, nw, k) == pytest.approx(spw.dpss(n, nw, Kmax=k, sym=True, norm=2), abs=1e-9)


def test_dpss_signs_orthonormality_and_validation():
    h = H.dpss(64, 2, 4)
    assert h @ h.T == pytest.approx(np.eye(4), abs=1e-10)
    for k in range(4):
        assert h[k][::-1] == pytest.approx(h[k] if k % 2 == 0 else -h[k], abs=1e-9)
    assert h[0].sum() > 0 and h[2].sum() > 0
    for k in (1, 3):
        assert h[k][h[k] ** 2 > max(1e-7, 1.0 / 64)][0] > 0
    for bad in ((2, 0.4, 1), (1, 0.2, 1), (8, 2, 0), (8, 2, 9), (8, 0, 1), (8, 4, 1), (8, -1, 1), (8, 2, -1)):
        with pytest.raises(ValueError):
            H.dpss(*bad)
    assert H.dpss(3, 1, 3).shape == (3, 3) and H.dpss(8, 3.9, 8).shape == (8, 8) and H.dpss(4, 1.9, 1).shape == (1, 4)


def test_dpss_cache_returns_copies_and_is_cleared_after_sixteen_entries():
    a = H.dpss(32, 2, 2)
    a[0, 0] = 99.0
    assert H.dpss(32, 2, 2)[0, 0] != 99.0
    H._CACHE.clear()
    for n in range(8, 8 + 17):
        H.dpss(n, 1.0, 1)
    assert len(H._CACHE) == 17
    H.dpss(40, 1.0, 1)
    assert len(H._CACHE) == 1


def test_sturm_count_by_hand_and_inverse_iteration_finds_the_eigenvector():
    d = np.array([1.0, 2.0, 3.0])
    e2 = np.array([0.0, 0.0])                                            # Diagonalmatrix: Eigenwerte 1, 2, 3
    assert H._sturm_count(d, e2, np.array([0.5, 1.5, 2.5, 3.5])).tolist() == [0, 1, 2, 3]
    assert H._sturm_count(d, e2, np.array([1.0 + 1e-9, 3.0 - 1e-9])).tolist() == [1, 2]
    dd, ee = np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0])
    v = H._inverse_iteration(dd, ee, 2.0 + np.sqrt(2.0) + 1e-9, np.array([1.0, 0.3, 0.7]))
    assert abs(v[0]) == pytest.approx(0.5, abs=1e-6) and abs(v[1]) == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_lines_resolved_needs_two_distinct_maxima_and_a_dip():
    spec = np.array([0.0, 4.0, 1.0, 4.0, 0.0])
    assert H.lines_resolved(spec, 1, 1.0, 3.0)
    assert not H.lines_resolved(np.array([0.0, 4.0, 2.0, 4.0, 0.0]), 1, 1.0, 3.0)        # Einbruch gleich depth * Maximum zählt nicht
    assert not H.lines_resolved(np.array([0.0, 4.0, 3.0, 2.0, 0.0]), 1, 1.0, 3.0)
    assert not H.lines_resolved(np.array([0.0, 1.0, 9.0, 1.0, 0.0]), 1, 1.0, 3.0)        # ein Gipfel
    assert H.lines_resolved(np.array([0.0, 4.0, 1.0, 4.0, 0.0]), 1, 3.0, 1.0)            # Reihenfolge der Linien egal
