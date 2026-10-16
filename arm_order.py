"""Ordnungskriterien für AR-Modelle: AIC, AICc, BIC, FPE aus den Fehlervarianzen der Ordnungen 1..pmax. Reine Funktionen."""

import math

import numpy as np

CRITERIA = ("AIC", "AICc", "BIC", "FPE")


def criteria(n: int, err, orders=None) -> dict:
    """Kriterienwerte je Ordnung; err[i] = Fehlervarianz der Ordnung orders[i] (Standard 1..len(err)). Es muss n > p + 1 gelten.

    AIC = n ln s^2 + 2 p; AICc = n ln s^2 + 2 p n / (n - p - 1); BIC = n ln s^2 + p ln n; FPE = s^2 (n + p + 1) / (n - p - 1).
    """
    e = np.asarray(err, dtype=float)
    ps = np.arange(1, len(e) + 1) if orders is None else np.asarray(orders)
    if np.any(ps >= n - 1):
        raise ValueError("Ordnung zu groß für die Länge")
    ln = n * np.log(e)
    return {"AIC": ln + 2 * ps, "AICc": ln + 2 * ps * n / (n - ps - 1), "BIC": ln + ps * math.log(n), "FPE": e * (n + ps + 1) / (n - ps - 1)}


def select_order(values, orders=None) -> int:
    """Ordnung mit dem kleinsten Kriterienwert (bei Gleichstand die kleinere)."""
    v = np.asarray(values, dtype=float)
    ps = np.arange(1, len(v) + 1) if orders is None else np.asarray(orders)
    return int(ps[int(np.argmin(v))])


def pmax_for(n: int, cap: int = 20) -> int:
    """Größte Ordnung, die ein Kriterium noch prüft: höchstens cap und n / 3."""
    return max(1, min(cap, n // 3))
