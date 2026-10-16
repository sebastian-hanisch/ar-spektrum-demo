"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_x():
    """x = (1, 2, 3, 4), Ordnung 1 von Hand: r0 = 30 / 4 = 7,5, r1 = (2 + 6 + 12) / 4 = 5; Yule-Walker a = 2/3, sigma^2 = 7,5 (1 - 4/9) = 25/6;
    Burg und Kovarianzmethode a = 2 (2 + 6 + 12) / ((4 + 9 + 16) + (1 + 4 + 9)) = 40/43."""
    return np.array([1.0, 2.0, 3.0, 4.0])
