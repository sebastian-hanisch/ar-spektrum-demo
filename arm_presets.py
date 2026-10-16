"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle gezeichneten Regler sind in jeder Einstellung wirksam: Die feste Ordnung erscheint nur bei „fest“, die Regler der Linien nur bei Linien (Linie 1 und Stärke ab „eine“,
Abstand und Stärke der Linie 2 bei „zwei“). Ihre Werte stehen nicht im Zustand eines Reglers, sondern unter `*_kept`. Die Ordnung ist durch die Länge begrenzt
(höchstens 20 und n / 3): `clamp_order` hält den Zustand gültig.
"""

import random
from dataclasses import dataclass

import streamlit as st

import arm_constants as C
import arm_order as O


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "method_select": SettingSpec("m", str, C.DEFAULT_METHOD, choices=C.METHODS),
    "order_mode_select": SettingSpec("om", str, C.DEFAULT_ORDER_MODE, choices=C.ORDER_MODES),
    "order_kept": SettingSpec("p", int, C.DEFAULT_ORDER, 1, C.ORDER_MAX),
    "background_select": SettingSpec("bg", str, C.DEFAULT_BACKGROUND, choices=tuple(C.BACKGROUNDS)),
    "noise_select": SettingSpec("nz", str, C.DEFAULT_NOISE, choices=C.NOISE_CHOICES),
    "lines_select": SettingSpec("ln", str, C.DEFAULT_LINES, choices=C.LINES),
    "line_kept": SettingSpec("f", float, C.DEFAULT_LINE, C.LINE_MIN, C.LINE_MAX),
    "gap_kept": SettingSpec("d", float, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "rel_kept": SettingSpec("rel", int, C.DEFAULT_REL, C.REL_MIN, C.REL_MAX),
    "snr_kept": SettingSpec("snr", int, C.DEFAULT_SNR, C.SNR_MIN, C.SNR_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_select", "method": "method_select", "order_mode": "order_mode_select", "order": "order_kept", "background": "background_select", "noise": "noise_select",
               "lines": "lines_select", "line": "line_kept", "gap": "gap_kept", "rel": "rel_kept", "snr": "snr_kept", "seed": "seed_input"}


def clamp_order():
    """Feste Ordnung höchstens pmax(n)."""
    top = O.pmax_for(st.session_state["n_select"], C.ORDER_MAX)
    if st.session_state["order_kept"] > top:
        st.session_state["order_kept"] = top


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default
    clamp_order()


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]
    clamp_order()


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
