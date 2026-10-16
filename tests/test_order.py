"""Ordnungskriterien: Handrechnung, Auswahl, Grenzen."""

import math

import numpy as np
import pytest

import arm_order as O


def test_criteria_by_hand():
    n, err = 10, [2.0, 1.5]
    c = O.criteria(n, err)
    ln = [10 * math.log(2.0), 10 * math.log(1.5)]
    assert c["AIC"] == pytest.approx([ln[0] + 2, ln[1] + 4], abs=1e-12)
    assert c["AICc"] == pytest.approx([ln[0] + 2 * 1 * 10 / 8, ln[1] + 2 * 2 * 10 / 7], abs=1e-12)
    assert c["BIC"] == pytest.approx([ln[0] + math.log(10), ln[1] + 2 * math.log(10)], abs=1e-12)
    assert c["FPE"] == pytest.approx([2.0 * 12 / 8, 1.5 * 13 / 7], abs=1e-12)
    assert O.CRITERIA == ("AIC", "AICc", "BIC", "FPE")


def test_orders_argument_and_the_length_limit():
    c = O.criteria(20, [1.0, 1.0], orders=[3, 5])
    assert c["AIC"] == pytest.approx([6.0, 10.0], abs=1e-12)
    with pytest.raises(ValueError):
        O.criteria(5, [1.0, 1.0, 1.0, 1.0])                         # Ordnung 4 >= n - 1
    assert O.criteria(6, [1.0, 1.0, 1.0, 1.0])["AIC"].shape == (4,)                 # größte Ordnung n - 2 ist noch erlaubt


def test_select_order_minimum_and_ties():
    assert O.select_order([5.0, 3.0, 4.0]) == 2
    assert O.select_order([3.0, 3.0, 4.0]) == 1                        # Gleichstand: die kleinere Ordnung
    assert O.select_order([5.0, 3.0, 4.0], orders=[4, 7, 9]) == 7


def test_bic_penalises_more_than_aic():
    # n = 100: Eine Ordnung mehr kostet bei AIC 2, bei BIC ln 100 = 4,6; der Gewinn ist -n ln(Verhältnis der Fehlervarianzen)
    err = np.array([1.0, 0.95])
    assert O.select_order(O.criteria(100, err)["AIC"]) == 2           # Gewinn 5,1 gegen Strafe 2
    assert O.select_order(O.criteria(100, err)["BIC"]) == 2           # Gewinn 5,1 gegen Strafe 4,6
    err = np.array([1.0, 0.97])
    assert O.select_order(O.criteria(100, err)["AIC"]) == 2           # Gewinn 3,0 gegen Strafe 2
    assert O.select_order(O.criteria(100, err)["BIC"]) == 1           # Gewinn 3,0 gegen Strafe 4,6


def test_maximum_order():
    assert O.pmax_for(32) == 10 and O.pmax_for(64) == 20 and O.pmax_for(1024) == 20 and O.pmax_for(2) == 1 and O.pmax_for(100, cap=5) == 5
