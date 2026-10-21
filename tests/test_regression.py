"""Regression tests: fixed-seed items, scores and fits must not drift silently."""

import json
import math
from pathlib import Path

import pytest

from changedet.regression import snapshot

GOLDEN = json.loads((Path(__file__).parent / "golden" / "snapshot.json").read_text())


@pytest.fixture(scope="module")
def current():
    return snapshot()


def _assert_close(a, b, path="root"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, float) or isinstance(b, float):
        if math.isnan(a):
            assert math.isnan(b), path
        else:
            assert a == pytest.approx(b, rel=1e-9, abs=1e-12), path
    else:
        assert a == b, path


def test_item_digests(current):
    assert current["item_digests"] == GOLDEN["item_digests"]


def test_first_prompt_text(current):
    assert current["first_prompt"] == GOLDEN["first_prompt"]


def test_scores_and_fits(current):
    _assert_close(current["responders"], GOLDEN["responders"])
