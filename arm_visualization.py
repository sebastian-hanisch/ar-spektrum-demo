"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import arm_constants as C
import arm_estimators as E
from arm_evaluation import Analysis

EPS = 1e-30


def _db(v):
    return 10.0 * np.log10(np.maximum(np.asarray(v, dtype=float), EPS))


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _line_positions(a: Analysis) -> list[float]:
    s = a.settings
    out = []
    if s.lines in ("eine", "zwei"):
        out.append(s.f1_hz)
    if s.lines == "zwei":
        out.append(s.f2_hz)
    return out


def build_spectrum(a: Analysis, zoom: bool = False) -> go.Figure:
    """Spektren in dB über der Frequenz in Hz; zoom: Ausschnitt um die Linien, sonst bis zur Nyquist-Frequenz."""
    lines = _line_positions(a)
    if zoom and lines:
        pad = max(4.0 * C.FS / a.settings.n, 0.5 * (max(lines) - min(lines)))
        lo, hi = max(0.0, min(lines) - pad), min(C.FS / 2, max(lines) + pad)
    else:
        lo, hi = 0.0, C.FS / 2
    sel = (a.f_hz >= lo) & (a.f_hz <= hi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.fft[sel]), mode="lines", line=dict(color=C.COLOR_FFT, width=1), name="Periodogramm, Hann",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.welch[sel]), mode="lines", line=dict(color=C.COLOR_WELCH, width=1.5), name=f"Welch, Länge {a.welch_seg}",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.est[sel]), mode="lines", line=dict(color=C.COLOR_AR, width=2.5), name=f"AR({a.p}), {a.settings.method}",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.true_density[sel]), mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dot"), name="wahre Dichte des Rauschens",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    for pos in lines:
        if lo <= pos <= hi:
            fig.add_vline(x=pos, line=dict(color=C.COLOR_LINE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Frequenz (Hz)", range=[lo, hi])
    fig.update_yaxes(title_text="Leistungsdichte (dB)")
    return _fixed(fig, height=400)


def build_poles(a: Analysis) -> go.Figure:
    """Polebene: Einheitskreis, wahre Pole des Hintergrunds und geschätzte Pole."""
    t = np.linspace(0.0, 2.0 * np.pi, 361)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="gray", width=1), name="Einheitskreis", hoverinfo="skip"))
    if len(a.true_poles):
        fig.add_trace(go.Scatter(x=a.true_poles.real, y=a.true_poles.imag, mode="markers", marker=dict(size=13, color=C.COLOR_TRUE, symbol="circle-open", line=dict(width=3)),
                                 name="wahre Pole", hovertemplate="Radius %{customdata:.3f}<extra></extra>", customdata=np.abs(a.true_poles)))
    if len(a.est_poles):
        fig.add_trace(go.Scatter(x=a.est_poles.real, y=a.est_poles.imag, mode="markers", marker=dict(size=9, color=C.COLOR_AR, symbol="x"), name="geschätzte Pole",
                                 hovertemplate="Radius %{customdata:.3f}<extra></extra>", customdata=np.abs(a.est_poles)))
    fig.update_xaxes(title_text="Realteil", range=[-1.2, 1.2], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Imaginärteil", range=[-1.2, 1.2])
    return _fixed(fig, height=380)


def build_criteria(a: Analysis) -> go.Figure:
    """Kriterien über der Ordnung als Abstand zum jeweiligen Minimum (FPE als n ln(FPE / Minimum)); die gewählte Ordnung liegt bei null."""
    n = a.settings.n
    fig = go.Figure()
    for name, v in a.criteria.items():
        d = n * np.log(v / v.min()) if name == "FPE" else v - v.min()
        fig.add_trace(go.Scatter(x=a.orders, y=d, mode="lines+markers", name=f"{name} (wählt {a.chosen[name]})", hovertemplate=f"{name}, Ordnung %{{x}}: %{{y:.1f}}<extra></extra>"))
    fig.add_vline(x=a.p, line=dict(color=C.COLOR_LINE, width=1.2, dash="dash"))
    top = max(float(np.max(n * np.log(v / v.min()) if k == "FPE" else v - v.min())) for k, v in a.criteria.items())
    fig.update_xaxes(title_text="Ordnung p", dtick=1)
    fig.update_yaxes(title_text="Kriterium minus Minimum", range=[-0.05 * top, 1.05 * top])
    return _fixed(fig, height=340)


def build_short_data(rows) -> go.Figure:
    ns = [str(r["n"]) for r in rows]
    fig = go.Figure()
    for method in E.METHODS:
        fig.add_trace(go.Bar(x=ns, y=[r["methods"][method]["rmse_db"] for r in rows], name=method, hovertemplate="n = %{x}: %{y:.2f} dB<extra></extra>"))
    fig.add_trace(go.Bar(x=ns, y=[r["welch_rmse_db"] for r in rows], name="Welch, Länge n/4", hovertemplate="n = %{x}: %{y:.2f} dB<extra></extra>"))
    fig.update_xaxes(title_text="Länge n", type="category")
    fig.update_yaxes(title_text="Fehler des Spektrums (RMSE dB)", type="log")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=360)


def build_order(rows) -> go.Figure:
    labels = [f"{r['background']}, n = {r['n']}" for r in rows]
    fig = go.Figure()
    for c in ("AIC", "AICc", "BIC", "FPE"):
        fig.add_trace(go.Bar(x=labels, y=[100 * r["criteria"][c]["right"] for r in rows], name=c, hovertemplate="%{x}: %{y:.0f} %<extra></extra>"))
    fig.update_yaxes(title_text="wahre Ordnung getroffen (%)", range=[0, 105])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=420)


def build_best(rows) -> go.Figure:
    labels = [f"n = {r['n']}, {r['background']}" for r in rows]
    fig = go.Figure()
    for key, name in (("welch", "Welch (bestes L)"), ("multitaper", "Multitaper (bestes NW)"), ("ar_true", "AR Burg, wahre Ordnung"), ("ar_aic", "AR Burg, AIC"),
                      ("ar_bic", "AR Burg, BIC"), ("yw_true", "AR Yule-Walker, wahre Ordnung")):
        fig.add_trace(go.Bar(x=labels, y=[r[key] for r in rows], name=name, hovertemplate="%{x}: %{y:.2f} dB<extra></extra>"))
    fig.update_yaxes(title_text="Fehler des Spektrums (RMSE dB)", type="log")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=460)


def build_noise(rows) -> go.Figure:
    snrs = [r["snr"] for r in rows]
    fig = go.Figure()
    for p in rows[0]["orders"]:
        fig.add_trace(go.Scatter(x=snrs, y=[r["orders"][p]["rmse_db"] for r in rows], mode="lines+markers", name=f"AR, Ordnung {p}",
                                 hovertemplate=f"Ordnung {p}, %{{x}} dB: %{{y:.2f}} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=snrs, y=[r["welch_rmse_db"] for r in rows], mode="lines+markers", name="Welch", line=dict(color="gray", dash="dot"),
                             hovertemplate="Welch, %{x} dB: %{y:.2f} dB<extra></extra>"))
    fig.update_xaxes(title_text="weißes Rauschen gegen AR-Anteil (dB)", autorange="reversed")
    fig.update_yaxes(title_text="Fehler des Spektrums (RMSE dB)", rangemode="tozero")
    return _fixed(fig, height=340)


def build_resolution(rows) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=r["deltas"], y=[100 * v for v in r["shares"]], mode="lines+markers", name=r["label"], hovertemplate=r["label"] + ", %{x} Bins: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Abstand der Linien (Bins von f_s/n)")
    fig.update_yaxes(title_text="Anteil aufgelöst (%)", range=[-3, 103])
    return _fixed(fig, height=420)


def build_splitting(rows) -> go.Figure:
    labels = [f"{r['method']}, {r['snr']:.0f} dB" for r in rows]
    fig = go.Figure()
    for i, p in enumerate(rows[0]["orders"]):
        fig.add_trace(go.Bar(x=labels, y=[100 * r["shares"][i] for r in rows], name=f"Ordnung {p}", hovertemplate="%{x}: %{y:.0f} %<extra></extra>"))
    fig.update_yaxes(title_text="Läufe mit Zusatzgipfel (%)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=380)


def build_frequency(rows) -> go.Figure:
    ns = [str(r["n"]) for r in rows]
    fig = go.Figure()
    for method in E.METHODS:
        fig.add_trace(go.Bar(x=ns, y=[r["methods"][method]["rmse"] for r in rows], name=method + " (AR 2)", hovertemplate="n = %{x}: %{y:.3f} Bins<extra></extra>"))
    fig.add_trace(go.Bar(x=ns, y=[r["fft_rmse"] for r in rows], name="Spitze, Hann-Periodogramm", hovertemplate="n = %{x}: %{y:.3f} Bins<extra></extra>"))
    fig.update_xaxes(title_text="Länge n", type="category")
    fig.update_yaxes(title_text="Frequenzfehler RMSE (Bins von 1/n)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_variance(rows) -> go.Figure:
    fig = go.Figure()
    for n in dict.fromkeys(r["n"] for r in rows):
        sel = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["p"] for r in sel], y=[r["std_db"] for r in sel], mode="lines+markers", name=f"gemessen, n = {n}", hovertemplate="p = %{x}: %{y:.2f} dB<extra></extra>"))
        fig.add_trace(go.Scatter(x=[r["p"] for r in sel], y=[r["formula_db"] for r in sel], mode="lines", line=dict(dash="dot"), name=f"Formel, n = {n}",
                                 hovertemplate="p = %{x}: %{y:.2f} dB<extra></extra>"))
    fig.update_xaxes(title_text="Ordnung p", type="log", tickvals=[r["p"] for r in rows[:4]])
    fig.update_yaxes(title_text="Streuung des Logarithmus (dB)", rangemode="tozero")
    return _fixed(fig, height=340)
