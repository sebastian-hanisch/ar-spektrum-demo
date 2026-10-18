"""AR-Spektrum: ein Modell statt einer Mittelung - scharf bei kurzen Daten, solange das Modell stimmt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Reihe „Spektral- und Wavelet-Analyse“. Ein simuliertes Signal (autoregressiver Hintergrund mit bekannten Polen, Zusatzrauschen, bis zu zwei Linien) wird mit einem
AR-Modell angepasst; gezeigt werden Schätzer, Ordnungswahl, Pole, Auflösung und das Scheitern bei falschem Modell. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import arm_constants as C
import arm_estimators as E
import arm_evaluation as ev
import arm_order as O
from arm_evaluation import Settings, analyse, de
from arm_pdf_export import build_pdf
from arm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from arm_visualization import (
    build_best,
    build_criteria,
    build_frequency,
    build_noise,
    build_order,
    build_poles,
    build_resolution,
    build_short_data,
    build_spectrum,
    build_splitting,
    build_variance,
)

st.set_page_config(page_title="AR-Spektrum – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("📈 AR-Spektrum: ein Modell statt einer Mittelung")
st.markdown(
    """
Welch und Multitaper mitteln Spektralwerte. Das **AR-Spektrum** passt stattdessen ein **autoregressives Modell** an: Jeder Messwert wird aus den letzten *p* vorhergesagt,
$x[t] = \\sum_k a_k x[t-k] + e[t]$; aus den wenigen Koeffizienten folgt das Spektrum $\\sigma^2/|A(f)|^2$. Bei kurzen Daten und schmalen Gipfeln ist das viel schärfer als Welch oder Multitaper,
**solange das Modell stimmt**: Die Ordnung muss passen, und das Signal muss wirklich autoregressiv sein.

Dies ist die fünfte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf **Multitaper** und den Demos davor aufbauend; „AR“ als Prognosemodell einer Zeitreihe ist hier nicht gemeint.
Alles ist **simuliert**: Pole, Linien und die wahre Dichte sind bekannt. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das AR-Spektrum", expanded=True):
    st.markdown(
        """
1. **Modell:** Die Daten werden als Ausgang eines Filters aus weißem Rauschen gesehen. Die Pole des Filters liegen innerhalb des Einheitskreises; je näher am Rand, desto schärfer der Gipfel.
2. **Drei Wege zu den Koeffizienten:** **Yule-Walker** löst die Normalgleichungen mit der Autokovarianz (verzerrt, glatt), **Burg** minimiert Vorwärts- und Rückwärtsfehler stufenweise,
   die **Kovarianzmethode** löst ein Kleinste-Quadrate-Problem auf den Daten.
3. **Ordnung:** Zu klein: Gipfel fehlen. Zu groß: mehr Streuung und je nach Schätzer und Signal Zusatzgipfel. **AIC**, **AICc**, **BIC** und **FPE** bestrafen die Ordnung unterschiedlich stark.
4. **Fehlspezifikation:** Weißes Rauschen auf dem Signal ist kein AR-Prozess mehr; Sinuslinien brauchen viele Pole.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Modell**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Anzahl der Messwerte (Abtastrate 1000 Hz).")
    pmax = O.pmax_for(n, C.ORDER_MAX)
    if st.session_state["order_kept"] > pmax:
        st.session_state["order_kept"] = pmax
    method = st.radio("Schätzer", list(C.METHODS), key="method_select", help="Yule-Walker: verzerrte Autokovarianz. Burg: Rekursion über Reflexionskoeffizienten. Kovarianz: Kleinste Quadrate.")
    order_mode = st.selectbox("Ordnung", list(C.ORDER_MODES), key="order_mode_select", help="„fest“: Sie wählen p. Sonst wählt das Kriterium p aus 1 bis pmax (höchstens 20 und n/3).")
    if order_mode == "fest":
        order = st.slider("Ordnung p", 1, pmax, min(int(st.session_state["order_kept"]), pmax), help=f"Zahl der Pole; hier höchstens {pmax}.")
        st.session_state["order_kept"] = int(order)
    else:
        order = int(st.session_state["order_kept"])
    st.markdown("**Signal**")
    background = st.selectbox("Hintergrund", list(C.BACKGROUNDS), key="background_select",
                              help="Autoregressiver Hintergrund mit bekannten Polen. Weiß: Ordnung 0. Resonanz: Ordnung 2. Steil: Ordnung 4, 57 dB Spannweite. Zwei scharfe Resonanzen: Ordnung 4.")
    noise = st.select_slider("Weißes Zusatzrauschen (gegen den AR-Anteil)", options=list(C.NOISE_CHOICES), key="noise_select",
                             help="Fügt weißes Rauschen hinzu: Das Signal ist dann kein AR-Prozess mehr (Fehlspezifikation). „aus“: exakt autoregressiv.")
    lines = st.radio("Linien", list(C.LINES), key="lines_select", horizontal=True, help="Reine Töne (Sinuslinien) auf dem Hintergrund (Lage und Stärke bekannt).")
    if lines != "keine":
        line = st.slider("Linie 1 (Bruchteil der Abtastrate)", C.LINE_MIN, C.LINE_MAX, float(st.session_state["line_kept"]), step=C.LINE_STEP, format="%.4f",
                         help="Lage der ersten Linie; mal 1000 Hz ergibt die Frequenz.")
        st.session_state["line_kept"] = float(line)
        snr = st.slider("Linie 1 gegen Rauschen (dB je Abtastwert)", C.SNR_MIN, C.SNR_MAX, int(st.session_state["snr_kept"]),
                        help="Linienleistung gegen die Rauschvarianz 1. 10 dB: Die Linie ragt deutlich heraus; 60 dB: kaum Rauschen.")
        st.session_state["snr_kept"] = int(snr)
    else:
        line, snr = float(st.session_state["line_kept"]), int(st.session_state["snr_kept"])
    if lines == "zwei":
        gap = st.slider("Abstand der Linie 2 (Bins von f_s/n)", C.GAP_MIN, C.GAP_MAX, float(st.session_state["gap_kept"]), step=C.GAP_STEP,
                        help="Abstand der zweiten Linie in Vielfachen von f_s/n. Ein Hann-Periodogramm trennt sie sicher erst ab etwa 3 Bins (bei 2 Bins in rund der Hälfte der Läufe).")
        st.session_state["gap_kept"] = float(gap)
        rel = st.slider("Linie 2 gegen Linie 1 (dB)", C.REL_MIN, C.REL_MAX, int(st.session_state["rel_kept"]), help="0 dB: gleich stark.")
        st.session_state["rel_kept"] = int(rel)
    else:
        gap, rel = float(st.session_state["gap_kept"]), int(st.session_state["rel_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "method_select": method, "order_mode_select": order_mode, "order_kept": int(order), "background_select": background, "noise_select": noise,
                   "lines_select": lines, "line_kept": float(line), "gap_kept": float(gap), "rel_kept": int(rel), "snr_kept": int(snr), "seed_input": int(seed)})

settings = Settings(int(n), method, order_mode, int(order), background, noise, lines, float(line), float(gap), int(rel), int(snr), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ordnung", f"{a.p}", delta=("fest gewählt" if order_mode == "fest" else f"{order_mode} wählt {a.p}"), delta_color="off")
m2.metric("Größter Polradius", de(a.max_radius, 3), delta=(f"wahr {de(float(abs(a.true_poles).max()), 3)}" if len(a.true_poles) else "weißes Rauschen: keine Pole"), delta_color="off")
if a.rmse_db is not None:
    m3.metric("Fehler des Spektrums", f"{de(a.rmse_db, 2)} dB", delta=f"Welch {de(a.welch_rmse_db, 2)} dB, Periodogramm {de(a.fft_rmse_db, 2)} dB", delta_color="off")
else:
    m3.metric("Fehler des Spektrums", "–", delta="nur ohne Linien messbar", delta_color="off")
m4.metric("Rechenaufwand", f"{a.ops:,}".replace(",", "."), delta=f"{de(a.ops / a.ops_fft, 1)}-fach einer FFT der Länge n", delta_color="off", help="Komplexe Multiplikationen der Anpassung (grobe Zählung).")
st.plotly_chart(build_spectrum(a, zoom=False), width="stretch", key="spectrum_full")
st.caption("Blau: AR-Spektrum. Grau: Welch (Hann, 50 % Überlappung, Länge n/4, mindestens 16) und das Periodogramm mit Hann. Grün gepunktet: wahre Dichte des Rauschens (ohne die Linien), "
           "rot gestrichelt: wahre Lage der Linien. Die Peakhöhe eines AR-Spektrums ist nicht die Leistung einer Linie.")
if lines != "keine":
    st.plotly_chart(build_spectrum(a, zoom=True), width="stretch", key="spectrum_zoom")
    st.caption("Ausschnitt um die Linien.")
pc1, pc2 = st.columns(2)
pc1.plotly_chart(build_poles(a), width="stretch", key="poles")
pc2.plotly_chart(build_criteria(a), width="stretch", key="criteria")
st.caption("Links: Pole in der komplexen Ebene (grüner Kreis: wahr, blaues Kreuz: geschätzt). Rechts: Ordnungskriterien als Abstand zum Minimum; die gestrichelte Linie ist die verwendete Ordnung. "
           f"Streuung des logarithmierten Spektrums abseits der Gipfel nach der Faustformel (gilt für weißes Rauschen): ±{de(a.std_formula_db, 2)} dB; am Gipfel streut das Spektrum deutlich stärker.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="ar_spektrum.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was gewinnt das Modell, und was riskiert es?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption(f"Feste Zufallsströme (Seed {C.SWEEP_SEED} und folgende), unabhängig von den Reglern. Abtastrate {C.FS:g} Hz. Fehler sind RMSE von 10·log₁₀(Schätzung/Wahrheit) in dB.")

st.markdown("#### Kurze Daten, schmales Spektrum")
short = _experiment("short_data")
st.caption(f"AR(2) mit Polen bei 0,2 und Radius 0,98, wahre Ordnung, {C.EXP_SHORT_RUNS} Läufe je Länge.")
st.dataframe(pd.DataFrame([{"n": r["n"], **{f"{m}: Radius": de(r["methods"][m]["radius"], 3) for m in E.METHODS}, **{f"{m}: Gipfel (dB)": de(r["methods"][m]["peak_db"], 1) for m in E.METHODS},
                            **{f"{m}: Fehler (dB)": de(r["methods"][m]["rmse_db"], 2) for m in E.METHODS}, "Welch: Fehler (dB)": de(r["welch_rmse_db"], 2)} for r in short]),
             hide_index=True, width="stretch")
st.plotly_chart(build_short_data(short), width="stretch", key="exp_short")
kind, text = ev.verdict_short_data(short)
getattr(st, kind)(text)

st.markdown("#### Welche Ordnung?")
order_res = _experiment("order")
st.caption(f"Burg, höchste geprüfte Ordnung 20 (oder n/3), {C.EXP_ORDER_RUNS} Läufe; Anteil der Läufe, in denen das Kriterium die wahre Ordnung trifft, die mittlere gewählte Ordnung und der Fehler.")
rows = []
for r in order_res:
    row = {"Hintergrund": r["background"], "wahre Ordnung": r["ptrue"], "n": r["n"]}
    for c in O.CRITERIA:
        row[c] = f"{de(100 * r['criteria'][c]['right'], 0)} % (Ø {de(r['criteria'][c]['mean'], 1)}, {de(r['criteria'][c]['rmse_db'], 2)} dB)"
    row["wahre Ordnung: Fehler"] = de(r["rmse_true"], 2) + " dB"
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.plotly_chart(build_order(order_res), width="stretch", key="exp_order")
kind, text = ev.verdict_order(order_res)
getattr(st, kind)(text)

st.markdown("#### Beste gegen beste – mit Heimvorteil")
best = _experiment("best")
st.caption(f"{C.EXP_BEST_RUNS} Läufe; je Verfahren der beste Parameter (Welch: Länge 16 bis 512, Multitaper: NW 2 bis 12). **Die Daten sind exakt autoregressiv: Das AR-Verfahren hat einen Heimvorteil.**")
st.dataframe(pd.DataFrame([{"n": r["n"], "Hintergrund": r["background"], "wahre Ordnung": r["ptrue"], "Welch": de(r["welch"], 2), "Multitaper": de(r["multitaper"], 2),
                            "AR Burg, wahre Ordnung": de(r["ar_true"], 2), "AR Burg, AIC": de(r["ar_aic"], 2), "AR Burg, BIC": de(r["ar_bic"], 2),
                            "AR Yule-Walker, wahre Ordnung": de(r["yw_true"], 2)} for r in best]), hide_index=True, width="stretch")
st.plotly_chart(build_best(best), width="stretch", key="exp_best")
kind, text = ev.verdict_best(best)
getattr(st, kind)(text)

st.markdown("#### Wenn das Modell nicht stimmt: weißes Zusatzrauschen")
noise_res = _experiment("noise")
st.caption(f"Resonanz (Pol 0,97) plus weißes Rauschen, n = {C.EXP_NOISE_N}, Burg, {C.EXP_NOISE_RUNS} Läufe; Welch mit Länge {C.EXP_NOISE_WELCH_SEG}.")
st.dataframe(pd.DataFrame([{"Rauschen (dB)": de(r["snr"], 0), **{f"Ordnung {p}: Gipfel (dB)": de(r["orders"][p]["peak_db"], 1) for p in r["orders"]},
                            **{f"Ordnung {p}: Fehler (dB)": de(r["orders"][p]["rmse_db"], 2) for p in r["orders"]}, "Welch: Fehler (dB)": de(r["welch_rmse_db"], 2)} for r in noise_res]),
             hide_index=True, width="stretch")
st.plotly_chart(build_noise(noise_res), width="stretch", key="exp_noise")
kind, text = ev.verdict_noise(noise_res)
getattr(st, kind)(text)

st.markdown("#### Linien: Auflösung, Aufspaltung, Frequenz")
st.caption("Sinuslinien sind kein autoregressiver Prozess: Das Modell braucht viele Pole und kann Zusatzgipfel zeigen.")
res = _experiment("resolution")
st.caption(f"Zwei gleich starke Linien (10 dB je Linie) in weißem Rauschen, n = {C.EXP_RES_N}, {C.EXP_RES_RUNS} Läufe; aufgelöst heißt zwei Maxima mit einem Einbruch von mindestens 3 dB.")
st.plotly_chart(build_resolution(res), width="stretch", key="exp_resolution")
kind, text = ev.verdict_resolution(res)
getattr(st, kind)(text)
split = _experiment("splitting")
st.caption(f"Eine Linie, n = {C.EXP_SPLIT_N}, {C.EXP_SPLIT_RUNS} Läufe: Anteil mit einem Zusatzgipfel (über {C.EXP_SPLIT_LEVEL_DB:g} dB gegen den Hauptgipfel).")
st.plotly_chart(build_splitting(split), width="stretch", key="exp_splitting")
kind, text = ev.verdict_splitting(split)
getattr(st, kind)(text)
freq = _experiment("frequency")
st.caption(f"Frequenz einer Linie bei {C.EXP_FREQ_SNR:g} dB aus dem Polpaar von AR(2) gegen die Spitze des Hann-Periodogramms, {C.EXP_FREQ_RUNS} Läufe, zufällige Lage und Phase.")
st.plotly_chart(build_frequency(freq), width="stretch", key="exp_frequency")
kind, text = ev.verdict_frequency(freq)
getattr(st, kind)(text)

st.markdown("#### Streuung des AR-Spektrums")
var = _experiment("variance")
st.caption(f"Weißes Rauschen als wahres Modell (Ordnung 0), Burg, {C.EXP_VAR_RUNS} Läufe; gemessene Streuung gegen die Faustformel 10/ln 10 · √(2p/n).")
st.dataframe(pd.DataFrame([{"n": r["n"], "Ordnung": r["p"], "gemessen (dB)": de(r["std_db"], 2), "Formel (dB)": de(r["formula_db"], 2), "Bias (dB)": de(r["bias_db"], 2)} for r in var]),
             hide_index=True, width="stretch")
st.plotly_chart(build_variance(var), width="stretch", key="exp_variance")
kind, text = ev.verdict_variance(var)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Signal ist autoregressiv** | Weißes Zusatzrauschen oder Nullstellen im wahren Spektrum: Die Pole wandern nach innen, Gipfel sinken; je stärker das Modell verletzt ist, desto höher muss die Ordnung sein. | Unterraumverfahren |
| **Die Ordnung stimmt** | Zu klein: Gipfel fehlen. Zu groß: mehr Streuung und je nach Schätzer und Signal Zusatzgipfel. AIC überschätzt, BIC trifft meist. | Konfidenz und Falschalarm |
| **Yule-Walker mit verzerrter Autokovarianz** | Bei kurzen Daten und steilen Spektren liegen die Pole zu weit innen; der Fehler wird groß. | Burg, Kovarianzmethode |
| **Sinuslinien** | Linienaufspaltung bei Burg, Zusatzgipfel bei hoher Ordnung; die Peakhöhe ist nicht die Linienleistung. | Unterraumverfahren |
| **Stationär** | Ändert sich das Spektrum, mittelt das Modell über die ganze Aufzeichnung. | Spektrogramm |
| **Mittelwertfrei, reell** | Ein Mittelwert muss vorher abgezogen werden. | – |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $x[t] = \sum_{k=1}^{p} a_k x[t-k] + e[t]$, $e$ weißes Rauschen mit Varianz $\sigma^2$. Spektrum: $S(f) = \dfrac{\sigma^2/f_s}{|1 - \sum_k a_k e^{-2\pi i f k/f_s}|^2}$; die Pole sind die Nullstellen
von $z^p - \sum_k a_k z^{p-k}$.

**Yule-Walker.** Mit der verzerrten Autokovarianz $r(k) = \frac1n\sum_t x[t]x[t+k]$ lösen die Normalgleichungen $\sum_j a_j r(|k-j|) = r(k)$ ein Toeplitz-System; die Levinson-Durbin-Rekursion liefert
die Reflexionskoeffizienten $\kappa_m$ und $\sigma_m^2 = \sigma_{m-1}^2(1-\kappa_m^2)$.

**Burg.** $\kappa_m = \dfrac{2\sum_t f_{m-1}[t]\,b_{m-1}[t-1]}{\sum_t f_{m-1}[t]^2 + \sum_t b_{m-1}[t-1]^2}$ mit den Vorwärts- und Rückwärtsfehlern; es gilt $|\kappa_m| \le 1$, für nicht entartete Daten $|\kappa_m| < 1$ (stabiles Modell).

**Kovarianzmethode (modifiziert).** Kleinste Quadrate auf der Summe der Vorwärts- und Rückwärtsvorhersagefehler über $t = p..n-1$.

**Ordnungskriterien.** $\mathrm{AIC} = n\ln\hat\sigma_p^2 + 2p$, $\mathrm{AICc} = n\ln\hat\sigma_p^2 + 2pn/(n-p-1)$, $\mathrm{BIC} = n\ln\hat\sigma_p^2 + p\ln n$, $\mathrm{FPE} = \hat\sigma_p^2\,(n+p+1)/(n-p-1)$.

**Streuung.** Näherung: $\mathrm{Std}\big(10\log_{10}\hat S\big) \approx \tfrac{10}{\ln 10}\sqrt{2p/n}$.

**Quellen.** Lehrbuchwissen (Burg 1975; Marple 1987; Kay 1988; Akaike 1974; Schwarz 1978). Formeln hier per Simulation, Handrechnung und gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
