"""Randfälle: kleinste und größte Länge, höchste Ordnung, alle Schätzer, Ordnungsarten und Hintergründe, Linien an den Reglergrenzen."""

import numpy as np
import pytest

import arm_constants as C
import arm_evaluation as ev


@pytest.mark.parametrize("method", C.METHODS)
@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_method_and_background_runs(method, background):
    a = ev.analyse(ev.Settings(n=64, method=method, background=background, order_mode="BIC", noise="20 dB"))
    assert np.isfinite(a.est).all() and (a.est > 0).all() and np.isfinite(a.welch).all() and a.rmse_db > 0 and 1 <= a.p <= 20


@pytest.mark.parametrize("mode", C.ORDER_MODES)
def test_every_order_mode_runs_at_the_shortest_and_the_longest_record(mode):
    for n in (32, 1024):
        a = ev.analyse(ev.Settings(n=n, order_mode=mode, order=3))
        assert np.isfinite(a.est).all() and (a.p == 3 if mode == "fest" else a.p == a.chosen[mode])


def test_the_order_is_capped_by_the_length_and_by_twenty():
    assert ev.analyse(ev.Settings(n=32, order_mode="fest", order=20)).p == 10
    assert ev.analyse(ev.Settings(n=1024, order_mode="fest", order=20)).p == 20
    a = ev.analyse(ev.Settings(n=32, order_mode="AIC"))
    assert len(a.orders) == 10 and a.criteria["AIC"].shape == (10,)


def test_white_noise_background_has_no_true_poles_and_no_peak_error():
    a = ev.analyse(ev.Settings(background="weiß", order_mode="fest", order=1))
    assert len(a.true_poles) == 0 and a.peak_db is None and a.rmse_db is not None


def test_lines_switch_off_the_error_measures_and_set_the_line_measures():
    a0 = ev.analyse(ev.Settings(lines="keine"))
    a1 = ev.analyse(ev.Settings(lines="eine"))
    a2 = ev.analyse(ev.Settings(lines="zwei"))
    assert a0.rmse_db is not None and a0.extra_peaks is None and a0.resolved is None
    assert a1.rmse_db is None and a1.peak_db is None and a1.extra_peaks is not None and a1.resolved is None
    assert a2.rmse_db is None and a2.resolved is not None and a2.fft_resolved is not None and a2.extra_peaks is None


@pytest.mark.parametrize("line", [C.LINE_MIN, C.LINE_MAX])
@pytest.mark.parametrize("gap", [C.GAP_MIN, C.GAP_MAX])
def test_lines_at_the_slider_limits_stay_below_nyquist(line, gap):
    for n in (32, 1024):
        s = ev.Settings(n=n, line=line, gap=gap, lines="zwei")
        assert 0 < s.f1_hz < C.FS / 2 and 0 < s.f2_hz < C.FS / 2 and abs(s.f2_hz - s.f1_hz) == pytest.approx(gap * C.FS / n)
        assert np.isfinite(ev.analyse(s).est).all()


def test_extreme_levels_noise_and_seed_limits():
    for rel in (C.REL_MIN, C.REL_MAX):
        for snr in (C.SNR_MIN, C.SNR_MAX):
            assert np.isfinite(ev.analyse(ev.Settings(lines="zwei", rel=rel, snr=snr)).est).all()
    for noise in C.NOISE_CHOICES:
        assert np.isfinite(ev.analyse(ev.Settings(noise=noise)).est).all()
    for seed in (0, C.SEED_MAX):
        assert np.isfinite(ev.analyse(ev.Settings(seed=seed)).est).all()


def test_extra_noise_adds_the_white_density_and_white_background_adds_in_quadrature():
    poles, ar_sigma, white = ev.noise_spec("Resonanz", 20.0)
    assert ar_sigma > 0 and white == pytest.approx(0.1)
    assert ev.noise_spec("weiß", 0.0)[2] == pytest.approx(2.0 ** 0.5)
    assert ev.noise_spec("weiß")[2] == 1.0 and ev.noise_spec("Resonanz")[2] == 0.0


def test_count_extra_peaks_by_hand():
    assert ev._count_extra_peaks([0.0, 1.0, 0.0, 0.5, 0.0], -10.0) == 1          # Gipfel 1,0 und 0,5 (über -10 dB = 0,1): ein zusätzlicher
    assert ev._count_extra_peaks([0.0, 1.0, 0.0, 0.05, 0.0], -10.0) == 0         # 0,05 liegt unter 0,1
    assert ev._count_extra_peaks([0.0, 1.0, 0.0], -10.0) == 0
    assert ev._count_extra_peaks([1.0, 0.5, 0.2], -10.0) == 0                    # Randpunkte zählen nie


def test_amplitude_from_snr():
    assert ev.amplitude(0.0) == pytest.approx(2.0 ** 0.5) and ev.amplitude(20.0) == pytest.approx((200.0) ** 0.5)
