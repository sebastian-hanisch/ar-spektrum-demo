"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import arm_evaluation as ev


def make_analysis(**kw):
    z = np.zeros(1)
    base = dict(settings=ev.Settings(), p=2, a=z, sigma2=1.0, x=z, f_hz=z, est=z, welch=z, welch_seg=32, fft=z, true_density=z, orders=np.array([1, 2]), criteria={}, chosen={},
                est_poles=np.zeros(0), true_poles=np.zeros(0), max_radius=0.95, rmse_db=0.8, welch_rmse_db=2.0, fft_rmse_db=5.1, peak_db=-0.5, resolved=None, fft_resolved=None,
                extra_peaks=None, std_formula_db=0.8, ops=100, ops_fft=500)
    base.update(kw)
    return ev.Analysis(**base)


def test_main_verdict_without_lines_better_and_worse_than_welch():
    kind, text = ev.verdict_main(make_analysis())
    assert kind == "success"
    assert text == "Ordnung 2 (Burg); größter Polradius 0,950. Fehler des Spektrums 0,80 dB gegen 2,00 dB für Welch und 5,10 dB für das Periodogramm."
    kind, _ = ev.verdict_main(make_analysis(rmse_db=2.5))
    assert kind == "warning"
    kind, _ = ev.verdict_main(make_analysis(rmse_db=2.0))
    assert kind == "warning"                                           # Gleichstand zählt nicht als besser


def test_main_verdict_one_line_with_and_without_extra_peaks():
    s = ev.Settings(lines="eine")
    kind, text = ev.verdict_main(make_analysis(settings=s, extra_peaks=0))
    assert kind == "info" and "einen einzigen Gipfel" in text
    kind, text = ev.verdict_main(make_analysis(settings=s, extra_peaks=2))
    assert kind == "warning" and "2 **Zusatzgipfel**" in text


def test_main_verdict_two_lines_all_four_cases():
    s = ev.Settings(lines="zwei")
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=True, fft_resolved=False))
    assert kind == "success" and "**zwei Gipfel**" in text and "das Periodogramm trennt sie nicht" in text
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=True, fft_resolved=True))
    assert kind == "success" and text.endswith("**zwei Gipfel**.")
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=False, fft_resolved=True))
    assert kind == "warning" and "**verschmelzen**" in text and "das Periodogramm trennt sie" in text
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=False, fft_resolved=False))
    assert kind == "warning" and text.endswith("**verschmelzen** im AR-Spektrum.")


def test_short_data_verdict():
    rows = [{"n": 64, "welch_rmse_db": 6.08, "methods": {"Yule-Walker": {"radius": 0.9536, "rmse_db": 2.99}, "Burg": {"radius": 0.9706, "rmse_db": 1.56}}}]
    kind, text = ev.verdict_short_data(rows)
    assert kind == "warning"
    assert "mit Yule-Walker bei 0,954, mit Burg bei 0,971 (wahr 0,98); der Fehler des Spektrums beträgt 2,99 dB gegen 1,56 dB, Welch 6,08 dB." in text


def test_order_verdict_reads_resonance_at_256():
    crit = {"AIC": {"right": 0.68, "under": 0.0, "mean": 2.9}, "AICc": {"right": 0.7, "under": 0.0, "mean": 2.8}, "BIC": {"right": 0.98, "under": 0.01, "mean": 2.0},
            "FPE": {"right": 0.68, "under": 0.0, "mean": 2.9}}
    rows = [{"background": "Resonanz", "n": 64, "criteria": {k: {**v, "right": 0.1} for k, v in crit.items()}}, {"background": "Resonanz", "n": 256, "criteria": crit}]
    kind, text = ev.verdict_order(rows)
    assert kind == "info"
    assert "trifft BIC die Ordnung in 98 % der Läufe, AIC in 68 %, AICc in 70 %, FPE in 68 %; zu kleine Ordnungen kommen 1 % vor." in text and "(mittlere Ordnung 2,9)" in text


def test_best_verdict_uses_n128_steep_and_the_longest_record():
    rows = [{"n": 128, "background": "Steil", "welch": 3.25, "multitaper": 4.92, "ar_true": 1.02, "yw_true": 10.23}, {"n": 1024, "background": "Steil", "welch": 1.0, "multitaper": 1.0,
             "ar_true": 0.4, "yw_true": 4.65}, {"n": 128, "background": "Resonanz", "welch": 9.0, "multitaper": 9.0, "ar_true": 9.0, "yw_true": 9.0}]
    kind, text = ev.verdict_best(rows)
    assert kind == "warning"
    assert "mit Burg und wahrer Ordnung 1,02 dB, mit Welch 3,25 dB, mit dem Multitaper 4,92 dB; mit Yule-Walker 10,2 dB (n = 1024: 4,7 dB)" in text and "Heimvorteil" in text


def test_noise_verdict_reads_ten_decibels():
    rows = [{"snr": 20.0, "welch_rmse_db": 9.0, "orders": {2: {"peak_db": -1.0, "rmse_db": 9.0}, 10: {"peak_db": -1.0, "rmse_db": 9.0}}},
            {"snr": 10.0, "welch_rmse_db": 1.8, "orders": {2: {"peak_db": -4.97, "rmse_db": 2.56}, 10: {"peak_db": -0.5, "rmse_db": 0.92}}}]
    kind, text = ev.verdict_noise(rows)
    assert kind == "warning" and "bei Ordnung 2 um 5,0 dB, der Fehler steigt auf 2,56 dB (Welch 1,80 dB); Ordnung 10 hält 0,92 dB" in text


def test_resolution_verdict():
    rows = [{"label": "Burg p = 16", "deltas": [1.5, 4.0], "shares": [0.99, 1.0]}, {"label": "Periodogramm, Hann", "deltas": [1.5, 4.0], "shares": [0.345, 1.0]},
            {"label": "Burg p = 4", "deltas": [1.5, 4.0], "shares": [0.0, 0.1]}]
    kind, text = ev.verdict_resolution(rows)
    assert kind == "info" and "mit Ordnung 16 die Linien in 99 % der Läufe, das Hann-Periodogramm in 34 %" in text and "bei 4 Bins noch in 10 %" in text


def test_splitting_verdict_reads_sixty_decibels():
    rows = [{"snr": 60.0, "method": "Burg", "orders": [2, 4], "shares": [0.0, 0.18]}, {"snr": 60.0, "method": "Kovarianz", "orders": [4, 12], "shares": [0.0, 0.31]},
            {"snr": 20.0, "method": "Burg", "orders": [2, 4], "shares": [0.0, 0.9]}]
    kind, text = ev.verdict_splitting(rows)
    assert kind == "warning" and "Ordnung 4 zeigt Burg in 18 % der Läufe" in text and "Ordnung 12 in 31 %" in text


def test_frequency_verdict_reads_n64():
    rows = [{"n": 32, "fft_rmse": 0.9, "methods": {"Burg": {"rmse": 9.0}, "Yule-Walker": {"rmse": 9.0}}},
            {"n": 64, "fft_rmse": 0.0213, "methods": {"Burg": {"rmse": 0.1246}, "Yule-Walker": {"rmse": 0.2417}}}]
    kind, text = ev.verdict_frequency(rows)
    assert kind == "info" and "um 0,12 Bins (Burg) bzw. 0,24 Bins (Yule-Walker) daneben, die Spitze des Hann-Periodogramms nur um 0,02 Bins" in text


def test_variance_verdict_reads_n128_p8():
    rows = [{"n": 128, "p": 2, "std_db": 9.0, "formula_db": 9.0}, {"n": 128, "p": 8, "std_db": 1.627, "formula_db": 1.5355}]
    kind, text = ev.verdict_variance(rows)
    assert kind == "info" and "Ordnung 8 streut das logarithmierte Spektrum um 1,63 dB; die Faustformel 10/ln 10 · √(2p/n) ergibt 1,54 dB" in text


def test_settings_geometry():
    s = ev.Settings(n=32, order=20)
    assert s.pmax == 10
    assert ev.Settings(n=128, line=0.2, gap=3.0).f2_hz == 200.0 + 3.0 * 1000.0 / 128
    assert ev.Settings(n=128, line=0.43, gap=12.0).f2_hz == 430.0 - 12.0 * 1000.0 / 128
