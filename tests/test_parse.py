import pytest

from changedet.parse import parse_response


@pytest.mark.parametrize("text,expected", [
    ("CHANGED: color, size", {"color", "size"}),
    ("changed: Colour and LOCATION", {"color", "position"}),
    ("Some reasoning.\nCHANGED: number; shape.", {"count", "shape"}),
    ("**CHANGED:** none", set()),
    ("CHANGED: None.", set()),
    ("CHANGED:", set()),
    ("CHANGED: color\nOn reflection...\nCHANGED: size", {"size"}),  # last line wins
    ("CHANGED: texture, color", {"texture", "color"}),  # distractor parsed, ignored in scoring
])
def test_parse_valid(text, expected):
    assert parse_response(text) == frozenset(expected)


@pytest.mark.parametrize("text", [
    "The color changed.",           # no CHANGED line
    "CHANGED: color, brightness",   # unknown attribute
    "",
])
def test_parse_unparseable(text):
    assert parse_response(text) is None
