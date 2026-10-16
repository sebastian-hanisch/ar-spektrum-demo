"""Vergleichsverfahren (wortgleich aus den Demos zu Bartlett/Welch und Multitaper): Hann-Fenster, Welch, Slepian-Taper, Multitaper, Auflösungskriterium.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy.
"""

import math

import numpy as np

NAMES = ("Rechteck", "Hann", "Hamming", "Blackman")
COSINE_SUM = {
    "Hann": (0.5, 0.5),
    "Hamming": (0.54, 0.46),
    "Blackman": (0.42, 0.5, 0.08),
}


def window(name: str, n: int) -> np.ndarray:
    """Periodisches Fenster der Länge n."""
    if name == "Rechteck":
        return np.ones(n)
    if name in COSINE_SUM:
        m = np.arange(n)
        w = np.zeros(n)
        for k, a in enumerate(COSINE_SUM[name]):
            w += (-1) ** k * a * np.cos(2.0 * np.pi * k * m / n)
        return w
    raise ValueError(f"unbekanntes Fenster: {name}")


def n_segments(n: int, seg: int, hop: int) -> int:
    """Zahl der vollständigen Segmente der Länge seg im Abstand hop in einer Folge der Länge n."""
    if seg > n or hop < 1:
        raise ValueError("Segment länger als die Folge oder Schrittweite < 1")
    return (n - seg) // hop + 1


def hop_for(seg: int, overlap: float) -> int:
    """Schrittweite in Abtastwerten für den Überlappungsanteil overlap in [0, 1)."""
    return max(1, int(round(seg * (1.0 - overlap))))


def welch(x, seg: int, hop: int, w, fs: float, nfft: int | None = None):
    """Mittel der gefensterten Segment-Periodogramme (Leistungsdichte je Hz, zweiseitig, Bins 0 .. nfft / 2) und Zahl der Segmente.

    x darf mehrere Läufe haben (letzte Achse = Zeit). Skalierung je Segment: |X|^2 / (fs sum w^2); Rechteck ohne Überlappung ist Bartlett.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(w) != seg:
        raise ValueError("Fensterlänge ungleich Segmentlänge")
    n = x.shape[-1]
    k = n_segments(n, seg, hop)
    segs = np.lib.stride_tricks.sliding_window_view(x, seg, axis=-1)[..., ::hop, :]
    p = np.abs(np.fft.rfft(segs * w, nfft or seg, axis=-1)) ** 2 / (fs * (w ** 2).sum())
    return p.mean(axis=-2), k


def _tridiagonal(n: int, nw: float):
    t = np.arange(n)
    d = ((n - 1 - 2 * t) / 2.0) ** 2 * math.cos(2.0 * math.pi * nw / n)
    e = t[1:] * (n - t[1:]) / 2.0
    return d, e


def _sturm_count(d, e2, x):
    """Zahl der Eigenwerte kleiner als x (Sturm-Folge), für ein Feld von Werten x."""
    q = d[0] - x
    count = (q < 0).astype(int)
    for j in range(1, len(d)):
        q = np.where(q == 0.0, 1e-300, q)
        q = d[j] - x - e2[j - 1] / q
        count += q < 0
    return count


def _inverse_iteration(d, e, shift, seed_vector, steps: int = 3):
    n = len(d)
    dd = d - shift
    x = seed_vector
    for _ in range(steps):
        cp = np.empty(n)
        dp = np.empty(n)
        cp[0] = e[0] / dd[0]
        dp[0] = x[0] / dd[0]
        for t in range(1, n):
            den = dd[t] - e[t - 1] * cp[t - 1]
            cp[t] = e[t] / den if t < n - 1 else 0.0
            dp[t] = (x[t] - e[t - 1] * dp[t - 1]) / den
        y = np.empty(n)
        y[-1] = dp[-1]
        for t in range(n - 2, -1, -1):
            y[t] = dp[t] - cp[t] * y[t + 1]
        x = y / np.linalg.norm(y)
    return x


_CACHE: dict = {}


def dpss(n: int, nw: float, k: int) -> np.ndarray:
    """Die k Slepian-Taper der Länge n und Zeit-Bandbreite-Produkt nw als k x n (Energie 1, symmetrische Fassung).

    Vorzeichen nach Percival und Walden: gerade Taper haben positive Summe, ungerade beginnen mit einem positiven Buckel. k <= n.
    """
    if n < 3 or k < 1 or k > n or nw <= 0 or nw >= n / 2:
        raise ValueError("Ungültige Taper-Parameter")
    key = (n, float(nw), k)
    if key in _CACHE:
        return _CACHE[key].copy()
    d, e = _tridiagonal(n, nw)
    e2 = e * e
    off = np.r_[0.0, e] + np.r_[e, 0.0]
    lo, hi = float((d - off).min()), float((d + off).max())
    a = np.full(k, lo)
    b = np.full(k, hi)
    target = n - 1 - np.arange(k)                 # der j-größte Eigenwert hat n - 1 - j Eigenwerte unter sich
    for _ in range(60):
        m = 0.5 * (a + b)
        up = _sturm_count(d, e2, m) <= target
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    lam = 0.5 * (a + b)
    out = np.empty((k, n))
    rng = np.random.default_rng(12345)
    for j in range(k):
        out[j] = _inverse_iteration(d, e, lam[j] + 1e-9 * (hi - lo), rng.normal(size=n))
    thresh = max(1e-7, 1.0 / n)
    for j in range(k):
        if j % 2 == 0:
            if out[j].sum() < 0:
                out[j] = -out[j]
        else:
            first = out[j][out[j] ** 2 > thresh][0]
            if first < 0:
                out[j] = -out[j]
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[key] = out
    return out.copy()


def eigenspectra(x, h, nfft: int | None = None) -> np.ndarray:
    """Eigenspektren Y_k(f) = sum_t h_k[t] x[t] exp(-2 pi i f t): Läufe (letzte Achse = Zeit) x k x Bins."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.fft.rfft(x[..., None, :] * h, nfft or x.shape[-1], axis=-1)


def multitaper_equal(x, h, fs: float, nfft: int | None = None) -> np.ndarray:
    """Mittel der k Eigenspektren mit gleichen Gewichten: (1 / k) sum_k |Y_k|^2 / fs."""
    return (np.abs(eigenspectra(x, h, nfft)) ** 2).mean(axis=-2) / fs


def lines_resolved(spec, pad: int, f1_bins: float, f2_bins: float, depth: float = 0.5) -> bool:
    """Zwei Linien gelten als aufgelöst, wenn das Spektrum um jede der beiden Linienlagen (Fenster ± halber Abstand) ein Maximum hat und das Minimum zwischen den
    Maxima unter depth mal dem kleineren Maximum liegt (Einbruch von mindestens 3 dB bei depth = 0,5). Frequenzen in Bins, das Spektrum liegt auf dem Raster 1 / pad."""
    spec = np.asarray(spec, dtype=float)
    lo_bin, hi_bin = sorted((f1_bins, f2_bins))
    c1, c2 = int(round(lo_bin * pad)), int(round(hi_bin * pad))
    half = max(1, int(round((c2 - c1) / 2.0)))
    seg1 = spec[max(c1 - half, 0):c1 + half + 1]
    seg2 = spec[max(c2 - half, 0):c2 + half + 1]
    i1 = max(c1 - half, 0) + int(np.argmax(seg1))
    i2 = max(c2 - half, 0) + int(np.argmax(seg2))
    if i1 >= i2:
        return False
    return bool(spec[i1:i2 + 1].min() < depth * min(spec[i1], spec[i2]))
