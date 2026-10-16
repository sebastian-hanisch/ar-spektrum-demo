"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from arm_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus "}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("AR-Spektrum: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    lines = {"keine": "keine Linie", "eine": f"eine Linie bei {de(s.f1_hz, 1)} Hz ({s.snr} dB je Abtastwert)",
             "zwei": f"zwei Linien bei {de(s.f1_hz, 1)} und {de(s.f2_hz, 1)} Hz ({s.snr} dB, zweite {s.rel} dB)"}[s.lines]
    order = f"fest {a.p}" if s.order_mode == "fest" else f"{s.order_mode} wählt {a.p}"
    pdf.multi_cell(0, 5, _s(f"Folge der Länge n = {s.n}, Schätzer {s.method}, Ordnung: {order}, Hintergrund: {s.background}, Zusatzrauschen: {s.noise}, {lines}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [f"Gewählte Ordnung je Kriterium: " + ", ".join(f"{k} {v}" for k, v in a.chosen.items()) + ".",
           f"Größter Polradius {de(a.max_radius, 3)}; Fehlervarianz sigma^2 = {de(a.sigma2, 4)}.",
           f"Streuung des logarithmierten Spektrums nach der Faustformel: {de(a.std_formula_db, 2)} dB.",
           f"Rechenaufwand (komplexe Multiplikationen): {a.ops} (Anpassung), {a.ops_fft} (eine FFT der Länge n)."]
    if a.rmse_db is not None:
        out.append(f"Fehler des Spektrums gegen die wahre Dichte: AR {de(a.rmse_db, 2)} dB, Welch {de(a.welch_rmse_db, 2)} dB, Periodogramm {de(a.fft_rmse_db, 2)} dB.")
    if a.peak_db is not None:
        out.append(f"Fehler am Gipfel des Hintergrunds: {de(a.peak_db, 2)} dB.")
    if a.resolved is not None:
        out.append(f"Zwei Linien getrennt: AR {'ja' if a.resolved else 'nein'}, Periodogramm {'ja' if a.fft_resolved else 'nein'}.")
    if a.extra_peaks is not None:
        out.append(f"Zusatzgipfel neben der Linie: {a.extra_peaks}.")
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: ar-spektrum-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
