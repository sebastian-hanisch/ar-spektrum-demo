"""Konstanten: Regler-Grenzen, feste Annahmen, Hintergründe, Experimente, Farben und Presets (Kern: arm_estimators.py, arm_order.py, arm_compare.py)."""

import arm_estimators as E
import arm_order as O

FS = 1000.0                                  # Abtastrate in Hz (feste Annahme)
N_CHOICES = (32, 64, 128, 256, 512, 1024)
DEFAULT_N = 128
METHODS = E.METHODS
DEFAULT_METHOD = "Burg"
ORDER_MODES = ("fest",) + O.CRITERIA
DEFAULT_ORDER_MODE = "fest"
ORDER_MAX = 20
DEFAULT_ORDER = 2                            # Ordnung bei „fest“
PAD = 4                                      # Auffüllfaktor des Auswertegitters (nur Darstellung)
PAD_RES = 8                                  # Auffüllfaktor beim Auflösungsexperiment
NOISE_CHOICES = ("aus", "40 dB", "20 dB", "10 dB", "0 dB")
NOISE_DB = {"aus": None, "40 dB": 40.0, "20 dB": 20.0, "10 dB": 10.0, "0 dB": 0.0}
DEFAULT_NOISE = "aus"
LINES = ("keine", "eine", "zwei")
DEFAULT_LINES = "keine"
LINE_MIN, LINE_MAX, LINE_STEP, DEFAULT_LINE = 0.05, 0.43, 0.0005, 0.2         # Linie 1 als Bruchteil der Abtastrate
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = 0.5, 12.0, 0.25, 3.0                # Abstand der Linie 2 in Bins von f_s / n
REL_MIN, REL_MAX, DEFAULT_REL = -40, 0, 0                                      # Stärke der Linie 2 gegen Linie 1 in dB
SNR_MIN, SNR_MAX, DEFAULT_SNR = -10, 60, 10                                    # Linie 1 gegen Rauschvarianz 1, in dB je Abtastwert
SEED_MAX, DEFAULT_SEED = 999999, 38

# Pole (Frequenz als Bruchteil der Abtastrate, Radius) je Hintergrund; Varianz auf 1 normiert
BACKGROUNDS = {"weiß": (), "Resonanz": ((0.2, 0.97),), "Steil": ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9)),
               "Zwei scharfe Resonanzen": ((0.1, 0.998), (0.3, 0.998))}
DEFAULT_BACKGROUND = "Resonanz"

# Experimente (feste Ströme)
SWEEP_SEED = 500000
EXP_SHORT_POLES = ((0.2, 0.98),)
EXP_SHORT_NS = (32, 64, 128, 256, 1024)
EXP_SHORT_RUNS = 300
EXP_ORDER_NS = (64, 256, 1024)
EXP_ORDER_CASES = (("Resonanz", 2), ("Zwei scharfe Resonanzen", 4), ("Steil", 4))
EXP_ORDER_RUNS = 300
EXP_BEST_NS = (128, 256, 1024)
EXP_BEST_BACKGROUNDS = (("Resonanz", 2), ("Steil", 4), ("Zwei scharfe Resonanzen", 4))
EXP_BEST_SEGS = (16, 32, 64, 128, 256, 512)
EXP_BEST_NW = (2, 3, 4, 6, 8, 12)
EXP_BEST_RUNS = 100
EXP_NOISE_N = 512
EXP_NOISE_DB = (40.0, 20.0, 10.0, 0.0)
EXP_NOISE_ORDERS = (2, 4, 10)
EXP_NOISE_RUNS = 200
EXP_NOISE_WELCH_SEG = 128
EXP_RES_N = 64
EXP_RES_DELTAS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EXP_RES_RUNS = 200
EXP_RES_SNR = 10.0
EXP_RES_METHODS = (("Burg p = 4", "Burg", 4), ("Burg p = 6", "Burg", 6), ("Burg p = 8", "Burg", 8), ("Burg p = 16", "Burg", 16), ("Kovarianz p = 4", "Kovarianz", 4),
                   ("Kovarianz p = 8", "Kovarianz", 8), ("Yule-Walker p = 8", "Yule-Walker", 8), ("Periodogramm, Hann", None, 0), ("Multitaper NW = 2", "Multitaper", 2))
EXP_SPLIT_N = 32
EXP_SPLIT_SNR = (20.0, 40.0, 60.0)
EXP_SPLIT_ORDERS = (2, 4, 8, 12)
EXP_SPLIT_RUNS = 300
EXP_SPLIT_LEVEL_DB = -15.0
EXP_FREQ_NS = (32, 64, 128)
EXP_FREQ_RUNS = 600
EXP_FREQ_SNR = 20.0
EXP_VAR_NS = (128, 512)
EXP_VAR_ORDERS = (2, 4, 8, 16)
EXP_VAR_RUNS = 400

COLOR_AR = "rgba(76,120,168,0.95)"
COLOR_WELCH = "rgba(150,150,150,0.85)"
COLOR_FFT = "rgba(190,190,190,0.5)"
COLOR_TRUE = "#2F6B65"
COLOR_LINE = "#B03A2E"
COLOR_EXPECTED = "#E8833A"

PRESETS = {
    "Kurze Daten, schmales Spektrum": {"n": 64, "method": "Burg", "order_mode": "fest", "order": 4, "background": "Zwei scharfe Resonanzen", "noise": "aus", "lines": "keine",
                                       "line": 0.2, "gap": 3.0, "rel": 0, "snr": 10, "seed": 40},
    "Ordnung wählen": {"n": 256, "method": "Burg", "order_mode": "AIC", "order": 4, "background": "Zwei scharfe Resonanzen", "noise": "aus", "lines": "keine", "line": 0.2,
                       "gap": 3.0, "rel": 0, "snr": 10, "seed": 42},
    "Steiler Hintergrund": {"n": 128, "method": "Yule-Walker", "order_mode": "fest", "order": 4, "background": "Steil", "noise": "aus", "lines": "keine", "line": 0.2,
                            "gap": 3.0, "rel": 0, "snr": 10, "seed": DEFAULT_SEED},
    "Zusatzrauschen": {"n": 512, "method": "Burg", "order_mode": "fest", "order": 2, "background": "Resonanz", "noise": "10 dB", "lines": "keine", "line": 0.2,
                       "gap": 3.0, "rel": 0, "snr": 10, "seed": DEFAULT_SEED},
    "Zwei Linien": {"n": 64, "method": "Burg", "order_mode": "fest", "order": 16, "background": "weiß", "noise": "aus", "lines": "zwei", "line": 0.2, "gap": 2.0, "rel": 0,
                    "snr": 10, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Kurze Daten, schmales Spektrum": "Nur 64 Messwerte, zwei scharfe Resonanzen, wahre Ordnung 4: Burg liegt nahe an der Wahrheit, Yule-Walker und Welch liegen weit daneben (Schätzer umschalten).",
    "Ordnung wählen": "Zwei scharfe Resonanzen (wahre Ordnung 4), n = 256: Bei diesem Seed wählt AIC Ordnung 12, BIC die richtige 4; im Mittel trifft AIC die Ordnung nur in 70 % der Läufe, BIC in 99 %.",
    "Steiler Hintergrund": "57 dB Spannweite, Yule-Walker mit der wahren Ordnung 4: Die verzerrte Autokovarianz liefert schlechte Pole, das Spektrum liegt weit daneben. Mit Burg stimmt es.",
    "Zusatzrauschen": "Resonanz plus weißes Rauschen (10 dB gegen den AR-Anteil), Ordnung 2: Das Modell stimmt nicht mehr, der Gipfel sinkt ab. Ordnung 10 oder „BIC“ fängt es ab.",
    "Zwei Linien": "Zwei Sinuslinien im Abstand von 2 Bins, Ordnung 16 = n/4: Das AR-Spektrum trennt sie, das Hann-Periodogramm nicht (bei diesem Seed; bei 2 Bins trennt es in rund der Hälfte der Läufe). Mit Ordnung 4 verschmelzen sie.",
}
