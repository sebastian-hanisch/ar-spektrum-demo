# AR-Spektrum: ein Modell statt einer Mittelung – scharf bei kurzen Daten, solange das Modell stimmt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-ar-spektrum-demo.streamlit.app/)** -->

Fünfte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos
zum **Periodogramm**, zu **Fenstern**, zu **Bartlett und Welch** und zum **Multitaper**. Ein simuliertes Signal (autoregressiver Hintergrund mit bekannten Polen, optional weißes
Zusatzrauschen, bis zu zwei Sinuslinien) wird mit einem AR-Modell angepasst; drei Schätzer, vier Ordnungskriterien, Pole, Auflösung und das Scheitern bei falschem Modell werden gegen die Wahrheit gemessen.

**Namensfalle:** „AR“ als Prognosemodell einer Zeitreihe (Zeitreihen-Demos) ist hier nicht gemeint, es geht um das Spektrum.

## Warum dieses Problem

Welch und Multitaper (vorige Demos) mitteln Spektralwerte und verlieren bei kurzen Daten Auflösung. Das AR-Verfahren schätzt wenige Parameter statt vieler Werte und gewinnt viel – solange
das Modell stimmt. Die Demo zeigt, was es gewinnt (Faktor 3 bis 5 im Fehler) und was es riskiert (Ordnung, Zusatzrauschen, Sinuslinien, Yule-Walker bei steilen Spektren).

## Modell

Hintergrund weiß, Resonanz (Pol 0,97 bei 200 Hz), „Steil“ (Ordnung 4, 57 dB Spannweite) oder zwei scharfe Resonanzen (Pole 0,998 bei 100 und 300 Hz), Varianz auf 1 normiert; optional weißes
Zusatzrauschen (40 bis 0 dB unter dem AR-Anteil) und null bis zwei Sinuslinien. Schätzer: Yule-Walker (Levinson-Durbin auf der verzerrten Autokovarianz), Burg, modifizierte Kovarianzmethode,
alle selbst in numpy; scipy dient nur als Orakel in den Tests. Die Abtastrate ist fest 1000 Hz.

## Methodik

n = 32 bis 1024, Ordnung fest (1 bis 20, höchstens n/3) oder per AIC, AICc, BIC, FPE gewählt. Vergleich mit Welch (Hann, 50 %, Länge n/4) und dem Hann-Periodogramm im Bild, in den Experimenten mit
dem besten Welch und dem besten Multitaper. Experimente über feste Zufallsströme (Seeds ab 500000), jede Zahl ein Mittel über 100 bis 600 Läufe. Fehler sind RMSE von 10·log₁₀(Schätzung/Wahrheit) in dB.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py` (Anteile bei 100 bis 600 Läufen: Standardfehler 2 bis 5 Prozentpunkte).

| Frage | Befund |
|---|---|
| Wie gut ist die Anpassung bei kurzen Daten (AR(2), Pole 0,98, wahre Ordnung, n = 64)? | Polradius 0,954 (Yule-Walker) gegen 0,971 (Burg), wahr 0,98; Fehler 2,99 / 1,56 / 1,52 dB, Welch 6,08 dB (Yule-Walker / Burg / Kovarianz). Mit n = 1024 liegen alle unter 0,5 dB. |
| Welche Ordnung trifft das Kriterium (wahre Ordnung 2, n = 256)? | AIC / AICc / BIC / FPE treffen in 68 / 70 / 98 / 68 % der Läufe; zu kleine Ordnungen kommen nie vor, AIC und FPE überschätzen. Bei n = 1024: BIC 99 bis 100 %. |
| Beste gegen beste (n = 128, Resonanz)? | 0,97 dB (AR) gegen 2,75 dB (Welch) und 2,52 dB (Multitaper). Bei steilem Hintergrund liegt der Fehler bei 10,2 dB mit Yule-Walker gegen 1,02 dB mit Burg (jeweils wahre Ordnung). |
| Und wenn das Modell nicht stimmt (weißes Rauschen 10 dB unter dem AR-Anteil, n = 512)? | Ordnung 2: Gipfel -5,0 dB, Fehler 2,56 dB gegen Welch 1,80 dB; Ordnung 4: 1,22 dB; Ordnung 10: 0,92 dB. |
| Wie dicht dürfen zwei Linien liegen (10 dB je Linie, n = 64, Einbruch mindestens 3 dB)? | Bei 1,5 Bins Abstand: Burg mit Ordnung 16 100 %, Hann-Periodogramm 34 % der Läufe; Burg mit der kleinsten Ordnung 4 trennt sie bei 4 Bins noch nie. |
| Zeigt das AR-Spektrum Zusatzgipfel (eine Linie, n = 32)? | Bei 60 dB und Ordnung 4: Burg in 18 % der Läufe (Linienaufspaltung); Kovarianzmethode bei Ordnung 12 in 31 %; Yule-Walker nie. |
| Wie genau ist die Frequenz einer Linie (20 dB, n = 64)? | 0,12 Bins (Burg), 0,02 Bins (Periodogramm): Für eine einzelne Linie ist die FFT genauer. |
| Streuung des AR-Spektrums? | Die Faustformel 10/ln 10 · √(2p/n) liegt 2 bis 18 % unter dem gemessenen Wert (n = 128, p = 8: 1,63 gegen 1,54 dB). |

## Ehrliche Grenzen

- **Heimvorteil:** Die Daten sind exakt autoregressiv; der Vergleich mit Welch und Multitaper zeigt die Obergrenze des Vorteils, nicht den Alltag. Zusatzrauschen und Sinuslinien sind das Gegengewicht.
- **Nur Prozesse bis Ordnung 4 aus Polen** (wie der Generator), keine Nullstellen (ARMA); keine Maximum-Entropie-Varianten, kein Prony.
- **Yule-Walker mit der üblichen verzerrten Autokovarianz** (Teilung durch n, ohne Fenster); eine andere Konvention verschiebt die Zahlen.
- **Die Auflösung hängt vom Einbruchskriterium, von 10 dB je Linie und von n = 64 ab**; die Ordnung n/4 ist die kleinste, mit der sich die Linien sicher trennen ließen, keine Regel.
- **Die Ordnungskriterien** nutzen die Fehlervarianz der Burg-Rekursion; bei kurzen Daten (n = 64) trifft BIC in 92 bis 94 %, nicht in 99 %.
- **Stationär, reell, mittelwertfrei.**

## Verifikation

- **Kern gegen Handrechnung:** x = (1, 2, 3, 4), Ordnung 1: Yule-Walker a = 2/3, σ² = 25/6; Burg und Kovarianzmethode a = 40/43; Levinson der Ordnung 2 aus r = (1; 0,5; 0,2); Ordnungskriterien aus σ².
- **Orakel (`tests/test_oracle_ar.py`):** Levinson gegen `scipy.linalg.solve_toeplitz` und die Normalgleichungen; Burg gegen eine unabhängige stufenweise Minimierung; Kovarianzmethode gegen die gestapelte
  Kleinste-Quadrate-Lösung; AR-Spektrum gegen `signal_generator.true_psd`; lange Reihe (n = 20 000) auf 2 % der wahren Koeffizienten; BIC trifft die Ordnung in über 90 %; Streuformel innerhalb 20 %;
  Welch gegen `scipy.signal.welch`, Taper gegen `scipy.signal.windows.dpss`.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Schätzer, Ordnungsarten, Hintergründe, Permalink, Ordnung gegen Länge, Footer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich) |
| `arm_estimators.py` | Autokovarianz, Levinson, Yule-Walker, Burg, Kovarianzmethode, AR-Spektrum, Pole, Streuformel |
| `arm_order.py` | Ordnungskriterien AIC, AICc, BIC, FPE |
| `arm_compare.py` | Welch, Taper, Multitaper, Auflösungskriterium (wortgleich aus den Vorgänger-Demos) |
| `arm_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Presets |
| `arm_evaluation.py` | Analyse, Experimente, Verdikte |
| `arm_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `arm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `arm_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Generator, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Unterraumverfahren (MUSIC, ESPRIT):** nutzen das Rauschmodell und die Linienzahl ausdrücklich; Thema der Folgedemo.
- **Konfidenzintervalle für AR-Spektren, Falschalarm:** hier nur die Faustformel der Streuung.
- **ARMA-Modelle, Prony, Maximum-Entropie-Varianten, Spektrogramm:** nicht enthalten.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf rund 45 Sekunden (danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
