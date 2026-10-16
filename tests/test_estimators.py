"""AR-Schätzer: Mini-Instanz von Hand, Levinson, Burg, Kovarianzmethode, Spektrum, Pole, Streuformel, Aufwand."""

import math

import numpy as np
import pytest

import arm_estimators as E


def test_autocovariance_by_a_loop(mini_x):
    r = E.autocovariance(mini_x, 3)
    assert r == pytest.approx([7.5, 5.0, 2.75, 1.0], abs=1e-12)                      # 30 / 4, 20 / 4, (3 + 8) / 4, 4 / 4
    for k in range(4):
        assert r[k] == pytest.approx(sum(mini_x[t] * mini_x[t + k] for t in range(4 - k)) / 4.0, abs=1e-12)


def test_yule_walker_of_the_mini_instance(mini_x):
    a, err, refl = E.yule_walker(mini_x, 1)
    assert a == pytest.approx([2.0 / 3.0], abs=1e-12) and err == pytest.approx([7.5, 25.0 / 6.0], abs=1e-12) and refl == pytest.approx([2.0 / 3.0], abs=1e-12)


def test_burg_and_covariance_of_the_mini_instance(mini_x):
    a, err, refl = E.burg(mini_x, 1)
    assert a == pytest.approx([40.0 / 43.0], abs=1e-12) and refl == pytest.approx([40.0 / 43.0], abs=1e-12)
    assert err[0] == pytest.approx(7.5) and err[1] == pytest.approx(7.5 * (1 - (40.0 / 43.0) ** 2), abs=1e-12)
    a2, s2 = E.modified_covariance(mini_x, 1)
    assert a2 == pytest.approx([40.0 / 43.0], abs=1e-12)
    # Fehlervarianz: ((2 - a 1)^2 + (3 - a 2)^2 + (4 - a 3)^2 + (1 - a 2)^2 + (2 - a 3)^2 + (3 - a 4)^2) / 6
    k = 40.0 / 43.0
    ref = ((2 - k) ** 2 + (3 - 2 * k) ** 2 + (4 - 3 * k) ** 2 + (1 - 2 * k) ** 2 + (2 - 3 * k) ** 2 + (3 - 4 * k) ** 2) / 6.0
    assert s2 == pytest.approx(ref, abs=1e-12)


def test_levinson_second_order_by_hand():
    a, err, refl = E.levinson(np.array([1.0, 0.5, 0.2]), 2)
    k2 = (0.2 - 0.5 * 0.5) / 0.75
    assert refl == pytest.approx([0.5, k2], abs=1e-12)
    assert a == pytest.approx([0.5 - k2 * 0.5, k2], abs=1e-12) and a[0] == pytest.approx(0.53333333, abs=1e-8)
    assert err == pytest.approx([1.0, 0.75, 0.75 * (1 - k2 ** 2)], abs=1e-12)


def test_burg_reflection_coefficients_stay_inside_the_unit_interval_and_the_error_falls():
    x = np.random.default_rng(1).normal(size=64)
    a, err, refl = E.burg(x, 12)
    assert np.all(np.abs(refl) < 1.0) and np.all(np.diff(err) < 0) and len(a) == 12


def test_burg_of_an_all_zero_series_has_no_nan():
    a, err, refl = E.burg(np.zeros(20), 3)
    assert np.all(a == 0.0) and np.all(np.isfinite(err)) and np.all(refl == 0.0)


def test_fit_dispatch_and_errors_by_order():
    x = np.random.default_rng(2).normal(size=80)
    for method in E.METHODS:
        a, s2 = E.fit(method, x, 3)
        assert len(a) == 3 and s2 > 0
        errs = E.errors_by_order(method, x, 4)
        assert len(errs) == 4 and errs[2] == pytest.approx(E.fit(method, x, 3)[1], rel=1e-9)
    with pytest.raises(ValueError):
        E.fit("Prony", x, 2)


def test_ar_spectrum_of_white_noise_and_of_an_ar1():
    f = np.array([0.0, 100.0, 500.0])
    assert E.ar_spectrum([], 2.0, f, 1000.0) == pytest.approx([0.002, 0.002, 0.002], abs=1e-15)
    s = E.ar_spectrum([0.5], 1.0, f, 1000.0)
    assert s[0] == pytest.approx(1.0 / 1000.0 / 0.25, rel=1e-12) and s[2] == pytest.approx(1.0 / 1000.0 / 2.25, rel=1e-12)


def test_poles_of_known_polynomials():
    assert sorted(E.poles([0.5, 0.0]).real) == pytest.approx([0.0, 0.5], abs=1e-12)      # z^2 - 0,5 z
    p = E.poles([1.0, -0.5])                                                             # z^2 - z + 0,5: Radius sqrt(0,5)
    assert np.abs(p) == pytest.approx([math.sqrt(0.5)] * 2, abs=1e-12)
    assert len(E.poles([])) == 0


def test_std_formula_and_operation_counts():
    assert E.std_db_formula(128, 8) == pytest.approx(10.0 / math.log(10.0) * math.sqrt(16.0 / 128.0), abs=1e-12)
    assert E.ops_ar("Yule-Walker", 1024, 16) == 1024 * 16 + 256 and E.ops_ar("Burg", 1024, 16) == 2 * 1024 * 16
    assert E.ops_ar("Kovarianz", 1024, 16) == 2 * (1024 - 16) * 256 + 16 ** 3
