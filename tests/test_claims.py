"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

from pathlib import Path

import numpy as np
import pytest

import arm_constants as C
import arm_evaluation as ev
import arm_order as O
from sweeps_cache import get


def preset(name, **over):
    return ev.Settings(**{**C.PRESETS[name], **over})


def A(name, **over):
    return ev.analyse(preset(name, **over))


def row(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())][0]


def test_short_data_preset_burg_is_near_the_truth_and_yule_walker_and_welch_are_not():
    b, y = A("Kurze Daten, schmales Spektrum"), A("Kurze Daten, schmales Spektrum", method="Yule-Walker")
    assert b.settings.n == 64 and b.p == 4 and b.rmse_db < 1.5 and y.rmse_db > 4.0 and b.welch_rmse_db > 4.0


def test_order_preset_aic_overshoots_and_bic_hits_at_this_seed_and_the_shares_in_the_text():
    a = A("Ordnung wählen")
    assert a.chosen["AIC"] > 4 and a.chosen["BIC"] == 4 and a.p == a.chosen["AIC"]
    r = row(get("order"), background="Zwei scharfe Resonanzen", n=256)
    assert round(100 * r["criteria"]["AIC"]["right"]) == 70 and round(100 * r["criteria"]["BIC"]["right"]) == 99


def test_steep_preset_yule_walker_is_far_off_and_burg_is_not():
    y, b = A("Steiler Hintergrund"), A("Steiler Hintergrund", method="Burg")
    assert y.settings.n == 128 and y.p == 4 and y.rmse_db > 10 and b.rmse_db < 1.0 and y.welch_rmse_db < y.rmse_db


def test_noise_preset_the_peak_sinks_and_a_higher_order_or_bic_catches_it():
    a = A("Zusatzrauschen")
    assert a.settings.noise == "10 dB" and a.peak_db < -5.0 and a.rmse_db > a.welch_rmse_db
    assert A("Zusatzrauschen", order=10).rmse_db < a.welch_rmse_db
    bic = A("Zusatzrauschen", order_mode="BIC")
    assert bic.p > 2 and bic.rmse_db < a.welch_rmse_db


def test_two_lines_preset_order_sixteen_resolves_and_order_four_and_the_periodogram_do_not():
    a = A("Zwei Linien")
    assert a.p == 16 == a.settings.n // 4 and a.resolved is True and a.fft_resolved is False
    assert A("Zwei Linien", order=4).resolved is False


def test_short_data_claims():
    rows = get("short_data")
    r = row(rows, n=64)
    yw, burg, cov = (r["methods"][m] for m in ("Yule-Walker", "Burg", "Kovarianz"))
    assert yw["radius"] < burg["radius"] < 0.98 and yw["rmse_db"] > 1.8 * burg["rmse_db"] and burg["rmse_db"] == pytest.approx(cov["rmse_db"], rel=0.1)
    assert r["welch_rmse_db"] > 3.5 * burg["rmse_db"]
    for m in ("Yule-Walker", "Burg", "Kovarianz"):
        radii = [x["methods"][m]["radius"] for x in rows]
        assert radii == sorted(radii)                                          # der Radius nähert sich mit n der Wahrheit 0,98


def test_order_claims():
    rows = get("order")
    for r in rows:
        c = r["criteria"]
        assert all(v["under"] == 0.0 for v in c.values())                   # zu kleine Ordnungen kommen nie vor
        assert c["BIC"]["right"] > 0.9 and c["AIC"]["right"] < 0.76 and c["FPE"]["right"] < 0.76
        assert c["AIC"]["mean"] > r["ptrue"] and c["BIC"]["rmse_db"] < c["AIC"]["rmse_db"]
        assert c["BIC"]["rmse_db"] < 1.1 * r["rmse_true"] + 0.05


def test_best_against_best_pattern():
    rows = get("best")
    for r in rows:
        assert r["ar_true"] < r["welch"] and r["ar_true"] < r["multitaper"] and r["ar_bic"] < r["ar_aic"] * 1.05
        assert 2.0 < min(r["welch"], r["multitaper"]) / r["ar_true"] < 6.0
        if r["background"] == "Steil":
            assert r["yw_true"] > 4.0 * r["ar_true"]


def test_noise_claims():
    rows = get("noise")
    r10 = row(rows, snr=10.0)
    assert r10["orders"][2]["peak_db"] < -4.5 and r10["orders"][2]["rmse_db"] > r10["welch_rmse_db"] and r10["orders"][10]["rmse_db"] < r10["welch_rmse_db"]
    r40 = row(rows, snr=40.0)
    assert r40["orders"][2]["rmse_db"] < r40["welch_rmse_db"] and r40["orders"][2]["peak_db"] > -1.0


def test_line_claims():
    res = {r["label"]: dict(zip(r["deltas"], r["shares"])) for r in get("resolution")}
    assert res["Burg p = 16"][1.5] > 0.95 and res["Periodogramm, Hann"][1.5] < 0.4 and res["Burg p = 4"][4.0] == 0.0 and res["Burg p = 4"][6.0] < 0.6
    split = {(r["method"], r["snr"]): dict(zip(r["orders"], r["shares"])) for r in get("splitting")}
    assert split[("Burg", 60.0)][4] > 0.1 and split[("Kovarianz", 60.0)][12] > 0.25 and all(v == 0.0 for k, d in split.items() if k[0] == "Yule-Walker" for v in d.values())
    freq = {r["n"]: r for r in get("frequency")}
    assert all(r["fft_rmse"] < 0.05 < r["methods"]["Burg"]["rmse"] for r in freq.values()) and freq[64]["methods"]["Yule-Walker"]["rmse"] > freq[64]["methods"]["Burg"]["rmse"]


def test_variance_formula_is_close_but_a_little_low():
    for r in get("variance"):
        assert r["formula_db"] <= r["std_db"] <= 1.2 * r["formula_db"]


def test_operation_counts_and_the_order_limit():
    a = ev.analyse(ev.Settings(n=1024, method="Burg", order_mode="fest", order=16))
    assert a.ops == 2 * 1024 * 16 and a.ops_fft == 5120 and a.p == 16
    assert ev.Settings(n=32).pmax == 10 and ev.analyse(ev.Settings(n=32, order_mode="fest", order=20)).p == 10


# --- README: die Zahlen der Befunde-Tabelle stehen im Text und stimmen mit den Messungen überein --------------------------------------------------


def _readme():
    return (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def test_readme_short_data_and_order_numbers():
    text = _readme()
    r = row(get("short_data"), n=64)
    m = r["methods"]
    assert f"Polradius {ev.de(m['Yule-Walker']['radius'], 3)} (Yule-Walker) gegen {ev.de(m['Burg']['radius'], 3)} (Burg)" in text
    assert f"Fehler {ev.de(m['Yule-Walker']['rmse_db'], 2)} / {ev.de(m['Burg']['rmse_db'], 2)} / {ev.de(m['Kovarianz']['rmse_db'], 2)} dB, Welch {ev.de(r['welch_rmse_db'], 2)} dB" in text
    o = row(get("order"), background="Resonanz", n=256)["criteria"]
    assert " / ".join(ev.de(100 * o[c]["right"], 0) for c in O.CRITERIA) + " %" in text
    o4 = [row(get("order"), background=bg, n=1024)["criteria"] for bg in ("Resonanz", "Zwei scharfe Resonanzen", "Steil")]
    assert f"BIC {ev.de(100 * min(c['BIC']['right'] for c in o4), 0)} bis {ev.de(100 * max(c['BIC']['right'] for c in o4), 0)} %" in text


def test_readme_best_noise_and_line_numbers():
    text = _readme()
    b = row(get("best"), n=128, background="Resonanz")
    s = row(get("best"), n=128, background="Steil")
    assert f"{ev.de(b['ar_true'], 2)} dB (AR) gegen {ev.de(b['welch'], 2)} dB (Welch) und {ev.de(b['multitaper'], 2)} dB (Multitaper)" in text
    assert f"{ev.de(s['yw_true'], 1)} dB mit Yule-Walker" in text
    n10 = row(get("noise"), snr=10.0)
    assert f"Gipfel {ev.de(n10['orders'][2]['peak_db'], 1)} dB, Fehler {ev.de(n10['orders'][2]['rmse_db'], 2)} dB gegen Welch {ev.de(n10['welch_rmse_db'], 2)} dB" in text
    res = {r["label"]: dict(zip(r["deltas"], r["shares"])) for r in get("resolution")}
    assert f"Burg mit Ordnung 16 {ev.de(100 * res['Burg p = 16'][1.5], 0)} %, Hann-Periodogramm {ev.de(100 * res['Periodogramm, Hann'][1.5], 0)} %" in text
    freq = row(get("frequency"), n=64)
    assert f"{ev.de(freq['methods']['Burg']['rmse'], 2)} Bins (Burg), {ev.de(freq['fft_rmse'], 2)} Bins (Periodogramm)" in text


def test_readme_variance_numbers():
    text = _readme()
    rows = get("variance")
    ratios = [r["std_db"] / r["formula_db"] - 1.0 for r in rows]
    assert f"liegt {ev.de(100 * min(ratios), 0)} bis {ev.de(100 * max(ratios), 0)} % unter dem gemessenen Wert" in text
    r = row(rows, n=128, p=8)
    assert f"n = 128, p = 8: {ev.de(r['std_db'], 2)} gegen {ev.de(r['formula_db'], 2)} dB" in text
