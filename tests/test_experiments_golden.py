"""Festwerte der Experimente (feste Ströme ab Seed 500000): fangen Vorzeichen, Exponenten, Normierungen und Lagen, die Toleranz-Tests übersehen.

Die Werte stammen aus dem ersten Lauf des geprüften Codes (Handrechnung und scipy-Orakel stehen in test_estimators.py, test_order.py und test_oracle_ar.py)."""

import pytest

import arm_constants as C
from sweeps_cache import get

TOL = dict(abs=2e-3)


def test_short_data_fixed_values():
    rows = get("short_data")
    assert [r["n"] for r in rows] == [32, 64, 128, 256, 1024]
    expect = {32: ((0.9267, -7.5064, 4.4888), (0.9621, -6.7281, 2.0759), (0.9626, -5.4998, 1.8558), 6.602),
              64: ((0.9536, -4.8372, 2.9915), (0.9706, -3.7738, 1.562), (0.9707, -3.4995, 1.5163), 6.0809),
              128: ((0.965, -3.3059, 1.8527), (0.974, -2.3686, 0.9887), (0.9741, -2.3714, 0.9881), 3.2578),
              256: ((0.9727, -1.8604, 1.123), (0.977, -1.2769, 0.6677), (0.977, -1.1839, 0.6531), 2.1676),
              1024: ((0.9781, -0.5453, 0.4228), (0.9792, -0.3444, 0.3412), (0.9792, -0.3362, 0.3397), 1.7531)}
    for r in rows:
        yw, burg, cov, welch = expect[r["n"]]
        got = r["methods"]
        assert (got["Yule-Walker"]["radius"], got["Yule-Walker"]["peak_db"], got["Yule-Walker"]["rmse_db"]) == pytest.approx(yw, **TOL)
        assert (got["Burg"]["radius"], got["Burg"]["peak_db"], got["Burg"]["rmse_db"]) == pytest.approx(burg, **TOL)
        assert (got["Kovarianz"]["radius"], got["Kovarianz"]["peak_db"], got["Kovarianz"]["rmse_db"]) == pytest.approx(cov, **TOL)
        assert r["welch_rmse_db"] == pytest.approx(welch, **TOL)


def test_order_fixed_values():
    rows = get("order")
    expect = {("Resonanz", 64): (1.4995, (0.6633, 3.3967, 2.1609), (0.75, 2.5733, 1.8456), (0.94, 2.08, 1.5762), (0.6667, 3.3567, 2.1508)),
              ("Resonanz", 256): (0.7118, (0.6833, 2.8967, 0.9315), (0.7, 2.7633, 0.9068), (0.9767, 2.0267, 0.7281), (0.6833, 2.8967, 0.9315)),
              ("Resonanz", 1024): (0.3271, (0.7233, 2.7867, 0.4321), (0.73, 2.7533, 0.4293), (0.99, 2.0133, 0.3327), (0.7233, 2.7867, 0.4321)),
              ("Zwei scharfe Resonanzen", 64): (1.9593, (0.5333, 7.1233, 2.9943), (0.72, 4.9267, 2.3636), (0.92, 4.1533, 2.0604), (0.56, 6.6467, 2.8942)),
              ("Zwei scharfe Resonanzen", 256): (0.8051, (0.7033, 4.97, 1.0137), (0.73, 4.8, 0.9878), (0.99, 4.0167, 0.815), (0.7033, 4.97, 1.0137)),
              ("Zwei scharfe Resonanzen", 1024): (0.4457, (0.7067, 5.1067, 0.5503), (0.7133, 5.04, 0.546), (0.9967, 4.0033, 0.4467), (0.7067, 5.1067, 0.5503)),
              ("Steil", 64): (1.555, (0.6133, 5.79, 2.3672), (0.74, 4.64, 1.9268), (0.9267, 4.1267, 1.6716), (0.6167, 5.7033, 2.341)),
              ("Steil", 256): (0.8132, (0.65, 5.0333, 1.0409), (0.6667, 4.8633, 1.0171), (0.9767, 4.0267, 0.8297), (0.65, 5.0333, 1.0409)),
              ("Steil", 1024): (0.4106, (0.6267, 5.2833, 0.5462), (0.64, 5.1633, 0.5382), (0.9867, 4.0133, 0.4158), (0.6267, 5.2833, 0.5462))}
    assert len(rows) == 9
    for r in rows:
        rmse_true, aic, aicc, bic, fpe = expect[(r["background"], r["n"])]
        assert r["rmse_true"] == pytest.approx(rmse_true, **TOL)
        for name, (right, mean, rmse) in zip(("AIC", "AICc", "BIC", "FPE"), (aic, aicc, bic, fpe)):
            c = r["criteria"][name]
            assert (c["right"], c["mean"], c["rmse_db"]) == pytest.approx((right, mean, rmse), **TOL)
            assert c["under"] == 0.0


def test_best_fixed_values():
    rows = get("best")
    expect = [(128, "Resonanz", 2.7508, 2.5168, 0.9703, 1.4398, 0.9937, 1.473), (128, "Steil", 3.252, 4.9234, 1.0207, 1.2744, 1.0282, 10.2281),
              (128, "Zwei scharfe Resonanzen", 5.506, 5.4531, 1.1497, 1.5484, 1.1855, 7.9538), (256, "Resonanz", 2.0132, 1.993, 0.7175, 1.015, 0.7431, 0.8613),
              (256, "Steil", 2.7122, 4.396, 0.8407, 1.0946, 0.8695, 8.5781), (256, "Zwei scharfe Resonanzen", 3.8702, 3.7906, 0.7869, 1.0143, 0.8018, 5.9273),
              (1024, "Resonanz", 1.215, 1.1354, 0.3202, 0.4641, 0.3246, 0.3467), (1024, "Steil", 1.4826, 2.2743, 0.415, 0.5453, 0.4193, 4.6513),
              (1024, "Zwei scharfe Resonanzen", 2.3113, 2.3034, 0.4465, 0.5439, 0.4498, 2.7636)]
    assert len(rows) == len(expect)
    for r, (n, bg, *vals) in zip(rows, expect):
        assert (r["n"], r["background"]) == (n, bg)
        assert [r["welch"], r["multitaper"], r["ar_true"], r["ar_aic"], r["ar_bic"], r["yw_true"]] == pytest.approx(vals, **TOL)


def test_noise_fixed_values():
    rows = get("noise")
    expect = {40.0: ({2: (0.9681, -0.5142, 0.4957), 4: (0.9681, -0.5187, 0.6203), 10: (0.9685, -0.5139, 0.8858)}, 1.7818),
              20.0: ({2: (0.9617, -1.2764, 0.8161), 4: (0.9674, -0.601, 0.6305), 10: (0.9683, -0.5128, 0.9044)}, 1.8002),
              10.0: ({2: (0.9079, -4.9686, 2.5575), 4: (0.9473, -2.331, 1.2155), 10: (0.9676, -0.561, 0.9167)}, 1.8007),
              0.0: ({2: (0.6352, -8.7398, 2.6853), 4: (0.7944, -6.4266, 2.0804), 10: (0.9435, -2.0476, 1.0622)}, 1.8133)}
    assert [r["snr"] for r in rows] == list(expect)
    for r in rows:
        orders, welch = expect[r["snr"]]
        assert r["welch_rmse_db"] == pytest.approx(welch, **TOL)
        for p, vals in orders.items():
            assert (r["orders"][p]["radius"], r["orders"][p]["peak_db"], r["orders"][p]["rmse_db"]) == pytest.approx(vals, **TOL)


def test_resolution_fixed_values():
    rows = get("resolution")
    expect = {"Burg p = 4": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.515], "Burg p = 6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.17, 0.91, 1.0], "Burg p = 8": [0.0, 0.0, 0.0, 0.0, 0.14, 0.99, 1.0, 1.0],
              "Burg p = 16": [0.015, 0.135, 0.67, 1.0, 1.0, 1.0, 1.0, 1.0], "Kovarianz p = 4": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
              "Kovarianz p = 8": [0.0, 0.0, 0.0, 0.0, 0.14, 0.99, 1.0, 1.0], "Yule-Walker p = 8": [0.0, 0.0, 0.0, 0.0, 0.0, 0.725, 1.0, 1.0],
              "Periodogramm, Hann": [0.055, 0.095, 0.205, 0.345, 0.455, 1.0, 1.0, 1.0], "Multitaper NW = 2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 1.0]}
    assert [r["label"] for r in rows] == list(expect)
    for r in rows:
        assert r["deltas"] == list(C.EXP_RES_DELTAS) and r["shares"] == pytest.approx(expect[r["label"]], abs=1e-3)


def test_splitting_fixed_values():
    rows = get("splitting")
    expect = {(20.0, "Yule-Walker"): [0.0, 0.0, 0.0, 0.0], (20.0, "Burg"): [0.0, 0.0, 0.0033, 0.0767], (20.0, "Kovarianz"): [0.0, 0.0, 0.0367, 0.3333],
              (40.0, "Yule-Walker"): [0.0, 0.0, 0.0, 0.0], (40.0, "Burg"): [0.0, 0.0267, 0.24, 0.2367], (40.0, "Kovarianz"): [0.0, 0.0, 0.04, 0.3133],
              (60.0, "Yule-Walker"): [0.0, 0.0, 0.0, 0.0], (60.0, "Burg"): [0.0, 0.18, 0.1833, 0.18], (60.0, "Kovarianz"): [0.0, 0.0, 0.0367, 0.31]}
    assert [(r["snr"], r["method"]) for r in rows] == list(expect)
    for r in rows:
        assert r["orders"] == [2, 4, 8, 12] and r["shares"] == pytest.approx(expect[(r["snr"], r["method"])], abs=1e-3)


def test_frequency_fixed_values():
    rows = get("frequency")
    expect = {32: ((0.1911, 0.164), (0.098, 0.0459), (0.0484, 0.0399), 0.0235), 64: ((0.2417, 0.2209), (0.1246, 0.0892), (0.0944, 0.0888), 0.0213),
              128: ((0.3436, 0.3276), (0.2106, 0.1904), (0.194, 0.1898), 0.0195)}
    for r in rows:
        yw, burg, cov, fft = expect[r["n"]]
        assert (r["methods"]["Yule-Walker"]["rmse"], r["methods"]["Yule-Walker"]["bias"]) == pytest.approx(yw, abs=2e-3)
        assert (r["methods"]["Burg"]["rmse"], r["methods"]["Burg"]["bias"]) == pytest.approx(burg, abs=2e-3)
        assert (r["methods"]["Kovarianz"]["rmse"], r["methods"]["Kovarianz"]["bias"]) == pytest.approx(cov, abs=2e-3)
        assert r["fft_rmse"] == pytest.approx(fft, abs=1e-3)


def test_variance_fixed_values():
    rows = get("variance")
    expect = [(128, 2, 0.8563, 0.7677, -0.1246), (128, 4, 1.1397, 1.0857, -0.1851), (128, 8, 1.6269, 1.5355, -0.3219), (128, 16, 2.2718, 2.1715, -0.6314),
              (512, 2, 0.4546, 0.3839, -0.0001), (512, 4, 0.5916, 0.5429, -0.0152), (512, 8, 0.7954, 0.7677, -0.048), (512, 16, 1.1124, 1.0857, -0.1155)]
    assert len(rows) == len(expect)
    for r, (n, p, std, formula, bias) in zip(rows, expect):
        assert (r["n"], r["p"]) == (n, p) and (r["std_db"], r["formula_db"], r["bias_db"]) == pytest.approx((std, formula, bias), abs=2e-3)
