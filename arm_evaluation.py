"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import arm_compare as H
import arm_constants as C
import arm_estimators as E
import arm_order as O
import signal_generator as G

DB = E.DB


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    method: str = C.DEFAULT_METHOD
    order_mode: str = C.DEFAULT_ORDER_MODE
    order: int = C.DEFAULT_ORDER
    background: str = C.DEFAULT_BACKGROUND
    noise: str = C.DEFAULT_NOISE
    lines: str = C.DEFAULT_LINES
    line: float = C.DEFAULT_LINE
    gap: float = C.DEFAULT_GAP
    rel: int = C.DEFAULT_REL
    snr: int = C.DEFAULT_SNR
    seed: int = C.DEFAULT_SEED

    @property
    def pmax(self) -> int:
        return O.pmax_for(self.n, C.ORDER_MAX)

    @property
    def f1_hz(self) -> float:
        return self.line * C.FS

    @property
    def f2_hz(self) -> float:
        """Lage der Linie 2: oberhalb von Linie 1; läge sie jenseits der Nyquist-Grenze (zwei Bins davor), dann unterhalb im selben Abstand."""
        step = self.gap * C.FS / self.n
        up = self.f1_hz + step
        return up if up <= C.FS / 2 - 2.0 * C.FS / self.n else self.f1_hz - step


def noise_spec(background: str, extra_db=None):
    """(Pole, Innovations-Sigma, weißes Sigma) eines Hintergrunds, auf Varianz 1 normiert; extra_db fügt weißes Rauschen mit dieser Leistung unter dem AR-Anteil hinzu."""
    poles = C.BACKGROUNDS[background]
    ar_sigma = G.ar_unit_sigma(G.ar_from_poles(poles)) if poles else 0.0
    white = 0.0 if poles else 1.0
    if extra_db is not None:
        white = math.sqrt(white ** 2 + 10.0 ** (-extra_db / 10.0))
    return poles, ar_sigma, white


def amplitude(snr_db: float) -> float:
    return math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


def true_poles(background: str) -> np.ndarray:
    """Pole des wahren AR-Hintergrunds (leer bei weißem Rauschen)."""
    poles = C.BACKGROUNDS[background]
    if not poles:
        return np.zeros(0, dtype=complex)
    return E.poles(G.ar_from_poles(poles))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    p: int                         # verwendete Ordnung
    a: np.ndarray                  # Koeffizienten
    sigma2: float
    x: np.ndarray
    f_hz: np.ndarray               # Auswertegitter
    est: np.ndarray                # AR-Spektrum
    welch: np.ndarray              # Welch, Hann, 50 %, Länge max(16, n / 4)
    welch_seg: int
    fft: np.ndarray                # Periodogramm, Hann
    true_density: np.ndarray       # wahre Dichte des Rauschens (ohne Linien)
    orders: np.ndarray             # 1 .. pmax
    criteria: dict                 # Kriterienwerte je Ordnung
    chosen: dict                   # je Kriterium gewählte Ordnung
    est_poles: np.ndarray
    true_poles: np.ndarray
    max_radius: float
    rmse_db: float | None          # Fehler des Spektrums gegen die wahre Dichte (nur ohne Linien)
    welch_rmse_db: float | None
    fft_rmse_db: float | None
    peak_db: float | None          # Fehler am Gipfel des Hintergrunds (nur mit Pol und ohne Linien)
    resolved: bool | None
    fft_resolved: bool | None
    extra_peaks: int | None        # Gipfel über -15 dB gegen den Hauptgipfel außer dem Hauptgipfel (bei einer Linie)
    std_formula_db: float
    ops: int
    ops_fft: int


def build_signal(s: Settings):
    rng = G.SplitMix64(s.seed)
    ph = 2.0 * math.pi * rng.uniform(2)
    poles, ar_sigma, white = noise_spec(s.background, C.NOISE_DB[s.noise])
    a1 = amplitude(s.snr)
    a2 = a1 * 10.0 ** (s.rel / 20.0)
    lines = []
    if s.lines in ("eine", "zwei"):
        lines.append((s.f1_hz, a1, float(ph[0])))
    if s.lines == "zwei":
        lines.append((s.f2_hz, a2, float(ph[1])))
    sig = G.make_signal(s.n, C.FS, lines, poles, ar_sigma, white, rng)
    return sig, (poles, ar_sigma, white)


def _rmse(est, true, pad: int) -> float:
    sl = slice(4 * pad, len(est) - 4 * pad)
    return float(math.sqrt(np.mean((DB * np.log(est[sl] / true[sl])) ** 2)))


def analyse(s: Settings) -> Analysis:
    sig, (poles, ar_sigma, white) = build_signal(s)
    n, pmax = s.n, s.pmax
    errs = E.errors_by_order(s.method, sig.x, pmax)
    orders = np.arange(1, pmax + 1)
    crit = O.criteria(n, errs, orders)
    chosen = {name: O.select_order(v, orders) for name, v in crit.items()}
    p = min(s.order, pmax) if s.order_mode == "fest" else chosen[s.order_mode]
    a, s2 = E.fit(s.method, sig.x, p)
    nfft = C.PAD * n
    f_hz = np.arange(nfft // 2 + 1) * C.FS / nfft
    est = E.ar_spectrum(a, s2, f_hz, C.FS)
    seg = max(16, n // 4)
    welch, _ = H.welch(sig.x, seg, H.hop_for(seg, 0.5), H.window("Hann", seg), C.FS, nfft)
    wn = H.window("Hann", n)
    fft = np.abs(np.fft.rfft(sig.x * wn, nfft)) ** 2 / (C.FS * (wn ** 2).sum())
    true = G.true_psd(f_hz, C.FS, poles, ar_sigma, white)
    est_poles = E.poles(a)
    radius = float(np.abs(est_poles).max())
    rmse = w_rmse = f_rmse = peak = None
    if s.lines == "keine":
        rmse, w_rmse, f_rmse = _rmse(est, true, C.PAD), _rmse(welch, true, C.PAD), _rmse(fft, true, C.PAD)
        if poles:
            k = int(round(poles[0][0] * nfft))
            peak = float(DB * math.log(est[k] / true[k]))
    resolved = fft_resolved = None
    if s.lines == "zwei":
        b1, b2 = s.f1_hz * n / C.FS, s.f2_hz * n / C.FS
        resolved = H.lines_resolved(est, C.PAD, b1, b2)
        fft_resolved = H.lines_resolved(fft, C.PAD, b1, b2)
    extra = None
    if s.lines == "eine":
        extra = _count_extra_peaks(est, C.EXP_SPLIT_LEVEL_DB)
    return Analysis(s, p, a, s2, sig.x, f_hz, est, welch, seg, fft, true, orders, crit, chosen, est_poles, true_poles(s.background), radius, rmse, w_rmse, f_rmse, peak,
                    resolved, fft_resolved, extra, E.std_db_formula(n, p), E.ops_ar(s.method, n, p), n // 2 * int(round(math.log2(n))))


def _count_extra_peaks(est, level_db: float) -> int:
    """Zahl der lokalen Maxima über level_db (gegen das Maximum) außer dem höchsten."""
    v = np.asarray(est, dtype=float)
    idx = [i for i in range(1, len(v) - 1) if v[i] > v[i - 1] and v[i] >= v[i + 1]]
    top = v.max()
    return max(0, sum(1 for i in idx if v[i] > top * 10.0 ** (level_db / 10.0)) - 1)


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def _batch(runs: int, n: int, poles, extra_db, seed: int):
    ar = G.ar_from_poles(poles) if poles else None
    ar_sigma = G.ar_unit_sigma(ar) if poles else 0.0
    white = 0.0 if poles else 1.0
    if extra_db is not None:
        white = math.sqrt(white ** 2 + 10.0 ** (-extra_db / 10.0))
    x = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(seed))
    f = np.arange(n // 2 + 1) / n
    true = G.true_psd(f, 1.0, poles, ar_sigma, white)                 # Abtastrate 1: Frequenzen als Bruchteil der Abtastrate
    return x, f, true


def _log_rmse(est, true) -> float:
    return float(math.sqrt(np.mean((DB * np.log(est / true)) ** 2)))


def _ar_est(method, x, p, f_frac):
    a, s2 = E.fit(method, x, p)
    return E.ar_spectrum(a, s2, f_frac, 1.0), a


def sweep_short_data(ns=C.EXP_SHORT_NS, runs: int = C.EXP_SHORT_RUNS) -> list[dict]:
    """Schmalbandiges AR(2) (Pole 0,2 / 0,98), wahre Ordnung: Polradius, Gipfelfehler und Fehler des Spektrums je Schätzer und Länge; Welch (Länge n / 4) zum Vergleich."""
    poles = C.EXP_SHORT_POLES
    ar = G.ar_from_poles(poles)
    sig = G.ar_unit_sigma(ar)
    rows = []
    for n in ns:
        x = G.make_noise_batch(runs, n, poles, sig, 0.0, G.SplitMix64(C.SWEEP_SEED + n))
        f = np.arange(4, n // 2 - 3) / n
        true = G.true_psd(f, 1.0, poles, sig, 0.0)
        s_peak = float(G.true_psd([poles[0][0]], 1.0, poles, sig, 0.0)[0])
        row = {"n": n, "runs": runs, "methods": {}}
        for method in E.METHODS:
            radius, peak, err = [], [], []
            for r in range(runs):
                a, s2 = E.fit(method, x[r], 2)
                radius.append(float(np.abs(E.poles(a)).max()))
                peak.append(DB * math.log(float(E.ar_spectrum(a, s2, [poles[0][0]])[0]) / s_peak))
                err.append(np.mean((DB * np.log(E.ar_spectrum(a, s2, f) / true)) ** 2))
            row["methods"][method] = {"radius": float(np.mean(radius)), "peak_db": float(np.mean(peak)), "rmse_db": float(math.sqrt(np.mean(err)))}
        seg = max(16, n // 4)
        est, _ = H.welch(x, seg, H.hop_for(seg, 0.5), H.window("Hann", seg), 1.0, n)
        row["welch_rmse_db"] = _log_rmse(est[:, 4:n // 2 - 3], true)
        rows.append(row)
    return rows


def sweep_order(ns=C.EXP_ORDER_NS, cases=C.EXP_ORDER_CASES, runs: int = C.EXP_ORDER_RUNS) -> list[dict]:
    """Ordnungswahl mit Burg: Anteil der richtigen und der zu kleinen Ordnung, mittlere Ordnung und Fehler des Spektrums je Kriterium; dazu der Fehler mit der wahren Ordnung."""
    rows = []
    for name, ptrue in cases:
        poles = C.BACKGROUNDS[name]
        for n in ns:
            x, f, true = _batch(runs, n, poles, None, C.SWEEP_SEED + 10 + n)
            sl = slice(4, n // 2 - 3)
            pmax = O.pmax_for(n, C.ORDER_MAX)
            orders = np.arange(1, pmax + 1)
            picks = {c: [] for c in O.CRITERIA}
            errs = {c: [] for c in O.CRITERIA}
            errs["wahr"] = []
            for r in range(runs):
                err = E.burg(x[r], pmax)[1][1:]
                crit = O.criteria(n, err, orders)
                for c in O.CRITERIA:
                    p = O.select_order(crit[c], orders)
                    picks[c].append(p)
                    est, _ = _ar_est("Burg", x[r], p, f[sl])
                    errs[c].append(np.mean((DB * np.log(est / true[sl])) ** 2))
                est, _ = _ar_est("Burg", x[r], ptrue, f[sl])
                errs["wahr"].append(np.mean((DB * np.log(est / true[sl])) ** 2))
            row = {"background": name, "ptrue": ptrue, "n": n, "runs": runs, "rmse_true": float(math.sqrt(np.mean(errs["wahr"]))), "criteria": {}}
            for c in O.CRITERIA:
                arr = np.array(picks[c])
                row["criteria"][c] = {"right": float(np.mean(arr == ptrue)), "under": float(np.mean(arr < ptrue)), "mean": float(arr.mean()),
                                      "rmse_db": float(math.sqrt(np.mean(errs[c])))}
            rows.append(row)
    return rows


def _usable_segs(n: int) -> list[int]:
    return [seg for seg in C.EXP_BEST_SEGS if seg <= n]


def _usable_nws(n: int) -> list:
    return [nw for nw in C.EXP_BEST_NW if 2 * nw < n // 4]


def sweep_best(ns=C.EXP_BEST_NS, backgrounds=C.EXP_BEST_BACKGROUNDS, runs: int = C.EXP_BEST_RUNS) -> list[dict]:
    """Beste gegen beste (RMSE in dB gegen die wahre Dichte): Welch, Multitaper, AR (Burg) mit wahrer Ordnung, mit AIC und mit BIC, Yule-Walker mit wahrer Ordnung.
    Der Heimvorteil ist offen: die Daten sind exakt autoregressiv."""
    rows = []
    for n in ns:
        for name, ptrue in backgrounds:
            poles = C.BACKGROUNDS[name]
            x, f, true = _batch(runs, n, poles, None, C.SWEEP_SEED + 20 + n)
            sl = slice(4, n // 2 - 3)

            def rmse(est):
                return _log_rmse(est[:, sl], true[sl])

            welch = min(rmse(H.welch(x, seg, H.hop_for(seg, 0.5), H.window("Hann", seg), 1.0, n)[0][:, :n // 2 + 1]) for seg in _usable_segs(n))
            mt = min(rmse(H.multitaper_equal(x, H.dpss(n, nw, 2 * nw - 1), 1.0, n)[:, :n // 2 + 1]) for nw in _usable_nws(n))
            pmax = O.pmax_for(n, C.ORDER_MAX)
            orders = np.arange(1, pmax + 1)
            acc = {"wahr": [], "AIC": [], "BIC": [], "yw": []}
            for r in range(runs):
                crit = O.criteria(n, E.burg(x[r], pmax)[1][1:], orders)
                picks = {"wahr": (ptrue, "Burg"), "AIC": (O.select_order(crit["AIC"], orders), "Burg"), "BIC": (O.select_order(crit["BIC"], orders), "Burg"),
                         "yw": (ptrue, "Yule-Walker")}
                for key, (p, method) in picks.items():
                    est, _ = _ar_est(method, x[r], p, f[sl])
                    acc[key].append(np.mean((DB * np.log(est / true[sl])) ** 2))
            rows.append({"n": n, "background": name, "ptrue": ptrue, "welch": welch, "multitaper": mt, "ar_true": float(math.sqrt(np.mean(acc["wahr"]))),
                         "ar_aic": float(math.sqrt(np.mean(acc["AIC"]))), "ar_bic": float(math.sqrt(np.mean(acc["BIC"]))), "yw_true": float(math.sqrt(np.mean(acc["yw"])))})
    return rows


def sweep_noise(n: int = C.EXP_NOISE_N, snrs=C.EXP_NOISE_DB, orders=C.EXP_NOISE_ORDERS, runs: int = C.EXP_NOISE_RUNS) -> list[dict]:
    """AR(2) (Pol 0,97) plus weißes Rauschen: Polradius, Gipfelfehler und Fehler des Spektrums je Ordnung (Burg); Welch zum Vergleich."""
    poles = C.BACKGROUNDS["Resonanz"]
    rows = []
    for snr in snrs:
        x, f, true = _batch(runs, n, poles, snr, C.SWEEP_SEED + 30)
        sl = slice(4, n // 2 - 3)
        k_peak = int(round(poles[0][0] * n))
        row = {"snr": snr, "runs": runs, "orders": {}}
        for p in orders:
            radius, peak, err = [], [], []
            for r in range(runs):
                a, s2 = E.fit("Burg", x[r], p)
                radius.append(float(np.abs(E.poles(a)).max()))
                est = E.ar_spectrum(a, s2, f, 1.0)
                peak.append(DB * math.log(est[k_peak] / true[k_peak]))
                err.append(np.mean((DB * np.log(est[sl] / true[sl])) ** 2))
            row["orders"][p] = {"radius": float(np.mean(radius)), "peak_db": float(np.mean(peak)), "rmse_db": float(math.sqrt(np.mean(err)))}
        seg = C.EXP_NOISE_WELCH_SEG
        est, _ = H.welch(x, seg, H.hop_for(seg, 0.5), H.window("Hann", seg), 1.0, n)
        row["welch_rmse_db"] = _log_rmse(est[:, :n // 2 + 1][:, sl], true[sl])
        rows.append(row)
    return rows


def _two_lines(runs: int, n: int, delta: float, snr: float, seed: int):
    rng = G.SplitMix64(seed)
    t = np.arange(n)
    sigma = math.sqrt(0.5 / 10.0 ** (snr / 10.0))
    f1 = (n / 4.5 + 2.0 * rng.uniform(runs)) / n
    f2 = f1 + delta / n
    ph = 2.0 * math.pi * rng.uniform(2 * runs).reshape(2, runs)
    x = (np.cos(2.0 * np.pi * f1[:, None] * t[None, :] + ph[0][:, None]) + np.cos(2.0 * np.pi * f2[:, None] * t[None, :] + ph[1][:, None])
         + sigma * rng.normal(runs * n).reshape(runs, n))
    return x, f1, f2


def sweep_resolution(n: int = C.EXP_RES_N, deltas=C.EXP_RES_DELTAS, runs: int = C.EXP_RES_RUNS, snr: float = C.EXP_RES_SNR) -> list[dict]:
    """Zwei gleich starke Sinuslinien (SNR je Linie) in weißem Rauschen: Anteil der Läufe mit zwei getrennten Maxima (Einbruch mindestens 3 dB) je Verfahren und Abstand (Bins von 1 / n)."""
    pad = C.PAD_RES
    fg = np.arange(pad * n // 2 + 1) / (pad * n)
    rows = [{"label": label, "shares": [], "deltas": list(deltas)} for label, _, _ in C.EXP_RES_METHODS]
    for d in deltas:
        x, f1, f2 = _two_lines(runs, n, d, snr, C.SWEEP_SEED + 40)
        for row, (_, method, p) in zip(rows, C.EXP_RES_METHODS):
            hits = 0
            for r in range(runs):
                if method is None:
                    est = np.abs(np.fft.rfft(x[r] * H.window("Hann", n), pad * n)) ** 2
                elif method == "Multitaper":
                    est = H.multitaper_equal(x[r], H.dpss(n, p, 2 * p - 1), 1.0, pad * n)
                else:
                    a, s2 = E.fit(method, x[r], p)
                    est = E.ar_spectrum(a, s2, fg, 1.0)
                hits += H.lines_resolved(est, pad, f1[r] * n, f2[r] * n)
            row["shares"].append(hits / runs)
    return rows


def sweep_splitting(n: int = C.EXP_SPLIT_N, snrs=C.EXP_SPLIT_SNR, orders=C.EXP_SPLIT_ORDERS, runs: int = C.EXP_SPLIT_RUNS) -> list[dict]:
    """Eine Sinuslinie: Anteil der Läufe, in denen das AR-Spektrum einen Zusatzgipfel (über -15 dB gegen den Hauptgipfel) zeigt, je Schätzer, Ordnung und SNR."""
    t = np.arange(n)
    fg = np.arange(8 * n // 2 + 1) / (8 * n)
    rows = []
    for snr in snrs:
        sigma = math.sqrt(0.5 / 10.0 ** (snr / 10.0))
        rng = G.SplitMix64(C.SWEEP_SEED + 50)
        f0 = (n / 4.5 + 2.0 * rng.uniform(runs)) / n
        ph = 2.0 * math.pi * rng.uniform(runs)
        x = np.cos(2.0 * np.pi * f0[:, None] * t[None, :] + ph[:, None]) + sigma * rng.normal(runs * n).reshape(runs, n)
        for method in E.METHODS:
            shares = []
            for p in orders:
                extra = 0
                for r in range(runs):
                    a, s2 = E.fit(method, x[r], p)
                    extra += _count_extra_peaks(E.ar_spectrum(a, s2, fg, 1.0), C.EXP_SPLIT_LEVEL_DB) > 0
                shares.append(extra / runs)
            rows.append({"snr": snr, "method": method, "orders": list(orders), "shares": shares, "runs": runs})
    return rows


def sweep_frequency(ns=C.EXP_FREQ_NS, runs: int = C.EXP_FREQ_RUNS, snr: float = C.EXP_FREQ_SNR) -> list[dict]:
    """Frequenz einer Sinuslinie aus AR(2) (Winkel des Polpaars) gegen die Spitze des Hann-Periodogramms: RMSE und Bias in Bins von 1 / n."""
    sigma = math.sqrt(0.5 / 10.0 ** (snr / 10.0))
    rows = []
    for n in ns:
        rng = G.SplitMix64(C.SWEEP_SEED + 60 + n)
        t = np.arange(n)
        f0 = (n // 8 + 0.1 + 0.8 * rng.uniform(runs)) / n
        ph = 2.0 * math.pi * rng.uniform(runs)
        x = np.cos(2.0 * np.pi * f0[:, None] * t[None, :] + ph[:, None]) + sigma * rng.normal(runs * n).reshape(runs, n)
        row = {"n": n, "runs": runs, "methods": {}}
        for method in E.METHODS:
            err = []
            for r in range(runs):
                a, _ = E.fit(method, x[r], 2)
                roots = E.poles(a)
                top = roots[int(np.argmax(np.abs(roots)))]
                err.append((abs(float(np.angle(top))) / (2.0 * math.pi) - f0[r]) * n)
            err = np.array(err)
            row["methods"][method] = {"rmse": float(math.sqrt(np.mean(err ** 2))), "bias": float(err.mean())}
        pk = np.abs(np.fft.rfft(x * H.window("Hann", n), 16 * n, axis=1)) ** 2
        fe = ((np.argmax(pk[:, 16:-16], axis=1) + 16) / (16.0 * n) - f0) * n
        row["fft_rmse"] = float(math.sqrt(np.mean(fe ** 2)))
        rows.append(row)
    return rows


def sweep_variance(ns=C.EXP_VAR_NS, orders=C.EXP_VAR_ORDERS, runs: int = C.EXP_VAR_RUNS) -> list[dict]:
    """Streuung des logarithmierten AR-Spektrums (Burg, weißes Rauschen als wahres Modell) gegen die Faustformel 10/ln 10 sqrt(2 p / n)."""
    rows = []
    for n in ns:
        x = G.make_noise_batch(runs, n, (), 0.0, 1.0, G.SplitMix64(C.SWEEP_SEED + 70))
        f = np.arange(8, n // 2 - 7) / n
        for p in orders:
            vals = np.array([DB * np.log(_ar_fit_spectrum(x[r], p, f)) for r in range(runs)])
            rows.append({"n": n, "p": p, "std_db": float(vals.std(axis=0).mean()), "formula_db": E.std_db_formula(n, p), "bias_db": float(vals.mean())})
    return rows


def _ar_fit_spectrum(x, p, f):
    a, s2 = E.fit("Burg", x, p)
    return E.ar_spectrum(a, s2, f, 1.0)


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    base = f"Ordnung {a.p} ({s.method}); größter Polradius {de(a.max_radius, 3)}."
    if s.lines == "zwei":
        if a.resolved:
            return "success", base + " Die beiden Linien erscheinen im AR-Spektrum als **zwei Gipfel**" + ("; das Periodogramm trennt sie nicht." if not a.fft_resolved else ".")
        return "warning", base + " Die beiden Linien **verschmelzen** im AR-Spektrum" + ("; das Periodogramm trennt sie." if a.fft_resolved else ".")
    if s.lines == "eine":
        if a.extra_peaks:
            return "warning", base + f" Das AR-Spektrum zeigt {a.extra_peaks} **Zusatzgipfel** neben der Linie (Linienaufspaltung bzw. zu hohe Ordnung)."
        return "info", base + " Das AR-Spektrum zeigt einen einzigen Gipfel."
    kind = "success" if a.rmse_db is not None and a.rmse_db < a.welch_rmse_db else "warning"
    return kind, base + (f" Fehler des Spektrums {de(a.rmse_db, 2)} dB gegen {de(a.welch_rmse_db, 2)} dB für Welch und {de(a.fft_rmse_db, 2)} dB für das Periodogramm.")


def _short_row(rows, n):
    return [r for r in rows if r["n"] == n][0]


def verdict_short_data(rows: list[dict]) -> tuple[str, str]:
    r = _short_row(rows, 64)
    m = r["methods"]
    return "warning", (f"Bei n = 64 liegt der Polradius mit Yule-Walker bei {de(m['Yule-Walker']['radius'], 3)}, mit Burg bei {de(m['Burg']['radius'], 3)} (wahr 0,98); der Fehler des Spektrums beträgt "
                       f"{de(m['Yule-Walker']['rmse_db'], 2)} dB gegen {de(m['Burg']['rmse_db'], 2)} dB, Welch {de(r['welch_rmse_db'], 2)} dB. **Die verzerrte Autokovarianz zieht die Pole "
                       "stärker nach innen als Burg und Kovarianz; auch deren Gipfel liegt noch zu niedrig.**")


def verdict_order(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["background"] == "Resonanz" and x["n"] == 256][0]
    c = r["criteria"]
    return "info", (f"Bei wahrer Ordnung 2 und n = 256 trifft BIC die Ordnung in {de(100 * c['BIC']['right'], 0)} % der Läufe, AIC in {de(100 * c['AIC']['right'], 0)} %, AICc in "
                    f"{de(100 * c['AICc']['right'], 0)} %, FPE in {de(100 * c['FPE']['right'], 0)} %; zu kleine Ordnungen kommen {de(100 * max(v['under'] for v in c.values()), 0)} % vor. "
                    f"**AIC und FPE überschätzen** (mittlere Ordnung {de(c['AIC']['mean'], 1)}).")


def verdict_best(rows: list[dict]) -> tuple[str, str]:
    top = max(r["n"] for r in rows)
    a = [r for r in rows if r["n"] == 128 and r["background"] == "Steil"][0]
    b = [r for r in rows if r["n"] == top and r["background"] == "Steil"][0]
    return "warning", (f"Bei n = 128 und steilem Hintergrund beträgt der Fehler mit Burg und wahrer Ordnung {de(a['ar_true'], 2)} dB, mit Welch {de(a['welch'], 2)} dB, mit dem Multitaper "
                       f"{de(a['multitaper'], 2)} dB; mit Yule-Walker {de(a['yw_true'], 1)} dB (n = {top}: {de(b['yw_true'], 1)} dB). **Die Daten sind exakt autoregressiv: Das ist der "
                       "Heimvorteil, nicht der Alltag.**")


def verdict_noise(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["snr"] == 10.0][0]
    o = r["orders"]
    return "warning", (f"Mit weißem Rauschen 10 dB unter dem AR-Anteil sinkt der Gipfel bei Ordnung 2 um {de(abs(o[2]['peak_db']), 1)} dB, der Fehler steigt auf {de(o[2]['rmse_db'], 2)} dB "
                       f"(Welch {de(r['welch_rmse_db'], 2)} dB); Ordnung 10 hält {de(o[10]['rmse_db'], 2)} dB. **Je stärker das Modell verletzt ist, desto höher muss die Ordnung sein.**")


def verdict_resolution(rows: list[dict]) -> tuple[str, str]:
    by = {r["label"]: dict(zip(r["deltas"], r["shares"])) for r in rows}
    return "info", (f"Bei 1,5 Bins Abstand trennt Burg mit Ordnung 16 die Linien in {de(100 * by['Burg p = 16'][1.5], 0)} % der Läufe, das Hann-Periodogramm in "
                    f"{de(100 * by['Periodogramm, Hann'][1.5], 0)} %; Burg mit der kleinsten Ordnung 4 trennt sie bei 4 Bins noch in {de(100 * by['Burg p = 4'][4.0], 0)} %. "
                    "**Hochauflösung unterhalb von 2 Bins gibt es nur mit hoher Ordnung (hier etwa 12 bis 16 bei n = 64).**")


def verdict_splitting(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["method"] == "Burg" and x["snr"] == 60.0][0]
    k = [x for x in rows if x["method"] == "Kovarianz" and x["snr"] == 60.0][0]
    return "warning", (f"Bei 60 dB und Ordnung 4 zeigt Burg in {de(100 * dict(zip(r['orders'], r['shares']))[4], 0)} % der Läufe einen Zusatzgipfel (Linienaufspaltung), die Kovarianzmethode bei "
                       f"Ordnung 12 in {de(100 * dict(zip(k['orders'], k['shares']))[12], 0)} %; Yule-Walker nie. **Eine Linie braucht Ordnung 2; höhere Ordnungen können bei Burg und Kovarianzmethode in bis zu etwa einem Drittel der Läufe Zusatzgipfel erzeugen.**")


def verdict_frequency(rows: list[dict]) -> tuple[str, str]:
    r = _short_row(rows, 64)
    return "info", (f"Für eine einzelne Linie bei 20 dB liegt die Frequenz aus AR(2) bei n = 64 um {de(r['methods']['Burg']['rmse'], 2)} Bins (Burg) bzw. {de(r['methods']['Yule-Walker']['rmse'], 2)} Bins "
                    f"(Yule-Walker) daneben, die Spitze des Hann-Periodogramms nur um {de(r['fft_rmse'], 2)} Bins (AR(2); die FFT-Zahl ist durch das Gitter nach oben begrenzt). **Für eine Linie ist die FFT genauer.**")


def verdict_variance(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["n"] == 128 and x["p"] == 8][0]
    return "info", (f"Bei n = 128 und Ordnung 8 streut das logarithmierte Spektrum um {de(r['std_db'], 2)} dB; die Faustformel 10/ln 10 · √(2p/n) ergibt {de(r['formula_db'], 2)} dB (gilt für weißes Rauschen als wahres Modell; am Gipfel streut das Spektrum deutlich stärker). "
                    "**Die Streuung wächst mit der Ordnung und fällt mit der Länge.**")
