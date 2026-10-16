"""Orakel (scipy und unabhängige Referenzen): Levinson, Burg, Kovarianzmethode, wahres Spektrum, Ordnungswahl, Streuformel, Vergleichsverfahren."""

import math

import numpy as np
import pytest
from scipy import linalg
from scipy import signal as sps
from scipy.signal import windows as spw

import arm_compare as H
import arm_constants as C
import arm_estimators as E
import arm_evaluation as ev
import arm_order as O
import signal_generator as G


def test_levinson_equals_solve_toeplitz_and_the_normal_equations():
    x = np.random.default_rng(1).normal(size=300)
    r = E.autocovariance(x, 6)
    a, err, _ = E.yule_walker(x, 6)
    assert a == pytest.approx(linalg.solve_toeplitz(r[:6], r[1:]), abs=1e-10)
    assert err[-1] == pytest.approx(r[0] - (a * r[1:]).sum(), abs=1e-10)           # sigma^2 = r0 - sum a_k r_k
    toeplitz = linalg.toeplitz(r[:6])
    assert toeplitz @ a == pytest.approx(r[1:], abs=1e-10)


def test_burg_equals_an_independent_stagewise_minimisation():
    x = np.random.default_rng(2).normal(size=40) + np.sin(0.7 * np.arange(40))
    a, err, refl = E.burg(x, 3)
    # unabhängige Fassung: je Stufe den Reflexionskoeffizienten durch direkte Minimierung der Summe der Vorwärts- und Rückwärtsfehlerquadrate bestimmen
    ef, eb = x.copy(), x.copy()
    ref = []
    for _ in range(3):
        f, b = ef[1:], eb[:-1]
        grid = np.linspace(-0.999, 0.999, 19981)
        cost = [((f - k * b) ** 2).sum() + ((b - k * f) ** 2).sum() for k in grid]
        k = float(grid[int(np.argmin(cost))])
        ref.append(k)
        ef, eb = f - k * b, b - k * f
    assert refl == pytest.approx(ref, abs=2e-4)


def test_modified_covariance_equals_a_stacked_least_squares():
    x = np.random.default_rng(3).normal(size=60)
    p = 3
    rows, rhs = [], []
    for t in range(p, 60):
        rows.append([x[t - k] for k in range(1, p + 1)])
        rhs.append(x[t])
    for t in range(0, 60 - p):
        rows.append([x[t + k] for k in range(1, p + 1)])
        rhs.append(x[t])
    ref, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    a, s2 = E.modified_covariance(x, p)
    assert a == pytest.approx(ref, abs=1e-10)
    assert s2 == pytest.approx(float(np.mean((np.array(rhs) - np.array(rows) @ ref) ** 2)), abs=1e-12)


def test_the_true_ar_spectrum_equals_the_generators_density():
    for background in ("Resonanz", "Steil", "Zwei scharfe Resonanzen"):
        poles = C.BACKGROUNDS[background]
        ar = G.ar_from_poles(poles)
        sig = G.ar_unit_sigma(ar)
        f = np.linspace(0.0, 500.0, 101)
        assert E.ar_spectrum(ar, sig ** 2, f, 1000.0) == pytest.approx(G.true_psd(f, 1000.0, poles, sig, 0.0), rel=1e-10)
    assert np.abs(E.poles(G.ar_from_poles(C.BACKGROUNDS["Resonanz"]))) == pytest.approx([0.97, 0.97], abs=1e-9)
    assert np.abs(ev.true_poles("Resonanz")) == pytest.approx([0.97, 0.97], abs=1e-9) and len(ev.true_poles("weiß")) == 0
    assert sorted(np.abs(ev.true_poles("Steil"))) == pytest.approx([0.9, 0.9, 0.99, 0.99], abs=1e-6)       # doppelter Pol 0,9: schlecht konditioniert


def test_long_series_recover_the_coefficients_for_all_three_methods():
    poles = ((0.2, 0.9),)
    ar = G.ar_from_poles(poles)
    x = G.make_noise_batch(1, 20000, poles, G.ar_unit_sigma(ar), 0.0, G.SplitMix64(1))[0]
    for method in E.METHODS:
        a, s2 = E.fit(method, x, 2)
        assert a == pytest.approx(ar, rel=0.02) and s2 == pytest.approx(G.ar_unit_sigma(ar) ** 2, rel=0.03)


def test_bic_finds_the_true_order_in_simulation():
    poles = C.BACKGROUNDS["Resonanz"]
    x = G.make_noise_batch(150, 256, poles, G.ar_unit_sigma(G.ar_from_poles(poles)), 0.0, G.SplitMix64(7))
    orders = np.arange(1, 21)
    hits = sum(O.select_order(O.criteria(256, E.burg(row, 20)[1][1:], orders)["BIC"], orders) == 2 for row in x)
    aic = sum(O.select_order(O.criteria(256, E.burg(row, 20)[1][1:], orders)["AIC"], orders) == 2 for row in x)
    assert hits / 150 > 0.9 and aic / 150 < 0.85


def test_the_spread_formula_is_within_twenty_percent_of_the_simulation():
    x = G.make_noise_batch(300, 256, (), 0.0, 1.0, G.SplitMix64(9))
    f = np.arange(8, 120) / 256.0
    for p in (2, 8):
        vals = np.array([E.DB * np.log(E.ar_spectrum(*E.fit("Burg", row, p), f, 1.0)) for row in x])
        assert vals.std(axis=0).mean() == pytest.approx(E.std_db_formula(256, p), rel=0.2)


def test_the_comparators_equal_scipy():
    x = G.SplitMix64(5).normal(1000)
    seg = 192
    w = H.window("Hann", seg)
    est, k = H.welch(x, seg, 96, w, 2.0)
    f, p = sps.welch(x, fs=2.0, window=w, nperseg=seg, noverlap=96, detrend=False, scaling="density", return_onesided=True)
    p = p.copy()
    p[1:-1] /= 2.0
    assert est == pytest.approx(p, rel=1e-10, abs=1e-14)
    ref, _ = spw.dpss(64, 3, Kmax=5, sym=True, norm=2, return_ratios=True)
    assert H.dpss(64, 3, 5) == pytest.approx(ref, abs=1e-9)


def test_the_analysis_uses_the_generators_density_as_truth():
    s = ev.Settings(n=128, background="Resonanz", noise="aus")
    a = ev.analyse(s)
    poles, ar_sigma, white = ev.noise_spec("Resonanz")
    assert a.true_density == pytest.approx(G.true_psd(a.f_hz, C.FS, poles, ar_sigma, white), rel=1e-12)
    # mit Zusatzrauschen kommt die weiße Dichte hinzu
    s2 = ev.Settings(n=128, background="Resonanz", noise="10 dB")
    a2 = ev.analyse(s2)
    assert a2.true_density == pytest.approx(a.true_density + 0.1 / C.FS, rel=1e-9)
