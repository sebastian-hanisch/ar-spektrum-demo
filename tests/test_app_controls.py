"""AppTest: Auswahlfelder, Permalink, Ordnung gegen Länge, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import arm_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


@pytest.mark.parametrize("method", C.METHODS)
def test_every_method_runs(method):
    at = _run()
    at.radio(key="method_select").set_value(method).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_background_runs(background):
    at = _run()
    at.selectbox(key="background_select").set_value(background).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("noise", C.NOISE_CHOICES)
def test_every_noise_level_runs(noise):
    at = _run()
    at.select_slider(key="noise_select").set_value(noise).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_the_fixed_order_slider_appears_only_for_a_fixed_order():
    at = _run({"om": "BIC"})
    assert "Ordnung p" not in _labels(at)
    at.selectbox(key="order_mode_select").set_value("fest").run(timeout=TIMEOUT)
    assert not at.exception and "Ordnung p" in _labels(at)
    at.selectbox(key="order_mode_select").set_value("AICc").run(timeout=TIMEOUT)
    assert not at.exception and "Ordnung p" not in _labels(at)


def test_the_fixed_order_survives_switching_the_mode_away_and_back():
    at = _run({"om": "fest", "p": "7"})
    assert [x for x in at.slider if x.label == "Ordnung p"][0].value == 7
    at.selectbox(key="order_mode_select").set_value("BIC").run(timeout=TIMEOUT)
    at.selectbox(key="order_mode_select").set_value("fest").run(timeout=TIMEOUT)
    assert [x for x in at.slider if x.label == "Ordnung p"][0].value == 7


def test_shortening_the_record_pulls_the_order_down():
    at = _run({"om": "fest", "p": "20", "n": "1024"})
    assert [x for x in at.slider if x.label == "Ordnung p"][0].value == 20
    at.select_slider(key="n_select").set_value(32).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    slider = [x for x in at.slider if x.label == "Ordnung p"][0]
    assert slider.value == 10 and slider.max == 10


def test_line_controls_appear_only_when_there_are_lines_and_keep_their_values():
    at = _run({"ln": "keine"})
    assert not any(label.startswith(("Linie 1", "Abstand", "Linie 2")) for label in _labels(at))
    at.radio(key="lines_select").set_value("eine").run(timeout=TIMEOUT)
    labels = _labels(at)
    assert not at.exception and "Linie 1 (Bruchteil der Abtastrate)" in labels and "Abstand der Linie 2 (Bins von f_s/n)" not in labels
    at2 = _run({"ln": "zwei", "f": "0.3", "d": "7.5", "rel": "-12", "snr": "-5"})
    at2.radio(key="lines_select").set_value("keine").run(timeout=TIMEOUT)
    at2.radio(key="lines_select").set_value("zwei").run(timeout=TIMEOUT)
    by = {x.label: x for x in at2.slider}
    assert by["Linie 1 (Bruchteil der Abtastrate)"].value == 0.3 and by["Abstand der Linie 2 (Bins von f_s/n)"].value == 7.5
    assert by["Linie 2 gegen Linie 1 (dB)"].value == -12 and by["Linie 1 gegen Rauschen (dB je Abtastwert)"].value == -5


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "512", "m": "Kovarianz", "om": "FPE", "p": "9", "bg": "Steil", "nz": "10 dB", "ln": "zwei", "f": "9", "d": "0", "rel": "-999", "snr": "999", "seed": "42"})
    assert at.select_slider(key="n_select").value == 512 and at.radio(key="method_select").value == "Kovarianz" and at.selectbox(key="order_mode_select").value == "FPE"
    assert at.selectbox(key="background_select").value == "Steil" and at.select_slider(key="noise_select").value == "10 dB" and at.radio(key="lines_select").value == "zwei"
    by = {x.label: x for x in at.slider}
    assert by["Linie 1 (Bruchteil der Abtastrate)"].value == C.LINE_MAX and by["Abstand der Linie 2 (Bins von f_s/n)"].value == C.GAP_MIN
    assert by["Linie 2 gegen Linie 1 (dB)"].value == C.REL_MIN and by["Linie 1 gegen Rauschen (dB je Abtastwert)"].value == C.SNR_MAX
    at2 = _run({"n": "3000", "m": "Prony", "om": "unsinn", "bg": "unsinn", "nz": "5 dB", "ln": "drei"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.radio(key="method_select").value == C.DEFAULT_METHOD
    assert at2.selectbox(key="order_mode_select").value == C.DEFAULT_ORDER_MODE and at2.select_slider(key="noise_select").value == C.DEFAULT_NOISE
    assert at2.radio(key="lines_select").value == C.DEFAULT_LINES


def test_permalink_with_too_high_an_order_is_clamped():
    at = _run({"n": "32", "om": "fest", "p": "20"})
    assert [x for x in at.slider if x.label == "Ordnung p"][0].value == 10


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Zwei Linien").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.radio(key="lines_select").value == "zwei" and at.select_slider(key="n_select").value == 64 and [x for x in at.slider if x.label == "Ordnung p"][0].value == 16


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("label", ["Abstand der Linie 2 (Bins von f_s/n)", "Linie 2 gegen Linie 1 (dB)"])
@pytest.mark.parametrize("edge", ["min", "max"])
def test_second_line_sliders_at_their_limits(label, edge):
    at = _run({"ln": "zwei"})
    slider = [x for x in at.slider if x.label == label][0]
    value = slider.min if edge == "min" else slider.max
    slider.set_value(int(value) if isinstance(slider.value, int) else value).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
