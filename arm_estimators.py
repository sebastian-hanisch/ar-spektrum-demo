"""Kern: AR-Schätzer (Yule-Walker, Burg, modifizierte Kovarianzmethode), AR-Spektrum, Pole, Fehlervarianz je Ordnung, Streuformel, Aufwand.

Konvention wie im Generator: x[t] = sum_k a_k x[t - k] + e[t], also A(z) = 1 - sum_k a_k z^-k. Reine Funktionen, kein Zufall, kein Streamlit; nur numpy.
"""

import math

import numpy as np

DB = 10.0 / math.log(10.0)
METHODS = ("Yule-Walker", "Burg", "Kovarianz")


def autocovariance(x, maxlag: int) -> np.ndarray:
    """Verzerrte Autokovarianz r(0 .. maxlag) = (1 / n) sum_t x[t] x[t + k] (ohne Mittelwertabzug, die Folge ist mittelwertfrei)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return np.correlate(x, x, mode="full")[n - 1:n + maxlag] / n


def levinson(r, p: int):
    """Levinson-Durbin: Koeffizienten a_1..a_p, Fehlervarianzen der Ordnungen 0..p und Reflexionskoeffizienten aus der Autokovarianz r."""
    a = np.zeros(0)
    err = [float(r[0])]
    refl = []
    for m in range(1, p + 1):
        acc = r[m] - (a * r[m - 1:0:-1]).sum() if m > 1 else r[1]
        k = acc / err[-1]
        a = np.r_[a - k * a[::-1], k] if m > 1 else np.array([k])
        err.append(err[-1] * (1.0 - k * k))
        refl.append(float(k))
    return a, np.array(err), np.array(refl)


def yule_walker(x, p: int):
    """Yule-Walker: Levinson auf der verzerrten Autokovarianz. Rückgabe wie `levinson`."""
    return levinson(autocovariance(x, p), p)


def burg(x, p: int):
    """Burg: Reflexionskoeffizient je Stufe aus der Summe der Vorwärts- und Rückwärtsfehlerquadrate. Rückgabe (a, Fehlervarianzen 0..p, Reflexionskoeffizienten)."""
    x = np.asarray(x, dtype=float)
    ef, eb = x.copy(), x.copy()
    a = np.zeros(0)
    err = [float((x ** 2).mean())]
    refl = []
    for m in range(1, p + 1):
        efp, ebp = ef[1:], eb[:-1]
        denom = (efp ** 2).sum() + (ebp ** 2).sum()
        k = 2.0 * (efp * ebp).sum() / denom if denom > 0 else 0.0
        a = np.r_[a - k * a[::-1], k] if m > 1 else np.array([k])
        ef, eb = efp - k * ebp, ebp - k * efp
        err.append(err[-1] * (1.0 - k * k))
        refl.append(float(k))
    return a, np.array(err), np.array(refl)


def modified_covariance(x, p: int):
    """Modifizierte Kovarianzmethode: Kleinste Quadrate auf den Vorwärts- und Rückwärtsvorhersagefehlern. Rückgabe (a, Fehlervarianz der Ordnung p)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rows_f = np.array([x[t - p:t][::-1] for t in range(p, n)])
    rows_b = np.array([x[t + 1:t + p + 1] for t in range(0, n - p)])
    mat = np.vstack([rows_f, rows_b])
    rhs = np.r_[x[p:], x[:n - p]]
    a, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
    res = rhs - mat @ a
    return a, float((res ** 2).mean())


def fit(method: str, x, p: int):
    """(a, Fehlervarianz) der Ordnung p mit dem gewählten Schätzer."""
    if method == "Yule-Walker":
        a, err, _ = yule_walker(x, p)
        return a, float(err[-1])
    if method == "Burg":
        a, err, _ = burg(x, p)
        return a, float(err[-1])
    if method == "Kovarianz":
        return modified_covariance(x, p)
    raise ValueError(f"unbekannter Schätzer: {method}")


def errors_by_order(method: str, x, pmax: int) -> np.ndarray:
    """Fehlervarianzen der Ordnungen 1..pmax (für die Ordnungskriterien). Yule-Walker und Burg aus einer Rekursion, die Kovarianzmethode aus je einer Anpassung."""
    if method == "Yule-Walker":
        return yule_walker(x, pmax)[1][1:]
    if method == "Burg":
        return burg(x, pmax)[1][1:]
    return np.array([modified_covariance(x, p)[1] for p in range(1, pmax + 1)])


def ar_spectrum(a, sigma2: float, f, fs: float = 1.0) -> np.ndarray:
    """Leistungsdichte sigma^2 / fs / |A(f)|^2 an den Frequenzen f (Hz), zweiseitig wie `signal_generator.true_psd`."""
    w = 2.0 * np.pi * np.asarray(f, dtype=float) / fs
    den = np.ones(len(w), dtype=complex)
    for k, ak in enumerate(a, start=1):
        den = den - ak * np.exp(-1j * w * k)
    return sigma2 / fs / np.abs(den) ** 2


def poles(a) -> np.ndarray:
    """Nullstellen von z^p - sum_k a_k z^(p - k): die Pole des Modells."""
    return np.roots(np.r_[1.0, -np.asarray(a, dtype=float)])


def std_db_formula(n: int, p: int) -> float:
    """Faustformel für die Streuung des logarithmierten AR-Spektrums: 10 / ln 10 * sqrt(2 p / n)."""
    return DB * math.sqrt(2.0 * p / n)


def ops_ar(method: str, n: int, p: int) -> int:
    """Komplexe Multiplikationen der Anpassung (grobe Zählung): Yule-Walker n p + p^2, Burg 2 n p, Kovarianz 2 (n - p) p^2 + p^3."""
    if method == "Yule-Walker":
        return n * p + p * p
    if method == "Burg":
        return 2 * n * p
    return 2 * (n - p) * p * p + p ** 3
