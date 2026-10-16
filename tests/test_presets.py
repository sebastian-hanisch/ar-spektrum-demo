"""Presets: Vollständigkeit, gültige Werte, Standardwerte; Ordnung höchstens pmax."""

import arm_constants as C
import arm_evaluation as ev
import arm_order as O
import arm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["method"] in C.METHODS and p["order_mode"] in C.ORDER_MODES and 1 <= p["order"] <= O.pmax_for(p["n"], C.ORDER_MAX), name
        assert p["background"] in C.BACKGROUNDS and p["noise"] in C.NOISE_CHOICES and p["lines"] in C.LINES
        assert C.LINE_MIN <= p["line"] <= C.LINE_MAX and C.GAP_MIN <= p["gap"] <= C.GAP_MAX and C.REL_MIN <= p["rel"] <= C.REL_MAX
        assert C.SNR_MIN <= p["snr"] <= C.SNR_MAX and 0 <= p["seed"] <= C.SEED_MAX


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.method in C.METHODS and s.order_mode in C.ORDER_MODES and s.background in C.BACKGROUNDS and s.noise in C.NOISE_CHOICES
    assert 1 <= s.order <= s.pmax


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_names_are_short_enough_for_the_buttons():
    assert max(len(n) for n in C.PRESETS) <= 32


def test_noise_choices_map_to_decibels():
    assert set(C.NOISE_DB) == set(C.NOISE_CHOICES) and C.NOISE_DB["aus"] is None and C.NOISE_DB["10 dB"] == 10.0


def test_second_line_never_leaves_the_band():
    for n in C.N_CHOICES:
        for line in (C.LINE_MIN, C.LINE_MAX):
            for gap in (C.GAP_MIN, 3.0, C.GAP_MAX):
                f2 = ev.Settings(n=n, line=line, gap=gap).f2_hz
                assert 0.0 < f2 <= C.FS / 2 - 2.0 * C.FS / n + 1e-9
