from collections import Counter

import numpy as np
import pytest

from changedet.attributes import DISTRACTORS, QUERYABLE
from changedet.generator import (LEVELS, Knobs, compute_ground_truth, generate_item,
                                 generate_set, items_digest)
from textdecode import decode_panel


def test_same_seed_same_items():
    assert items_digest(generate_set(3, 30, 7)) == items_digest(generate_set(3, 30, 7))


def test_different_seed_different_items():
    assert items_digest(generate_set(3, 30, 7)) != items_digest(generate_set(3, 30, 8))


def test_level_knobs_respected(items):
    for it in items:
        preset = LEVELS[it.level]
        k = it.knobs
        assert (k.n_objects, k.n_attrs, k.surface_noise, k.n_distractors) == (
            preset["n_objects"], preset["n_attrs"], preset["surface_noise"],
            preset["n_distractors"])
        assert len(it.queried) == k.n_attrs and len(it.distractors) == k.n_distractors
        assert len(it.panels) == k.n_panels
        assert all(len(p) == k.n_objects for p in it.panels)
        assert it.n_changed == k.n_changed


def test_n_changed_covers_full_range(items):
    # N is identifiable only if items with n_changed > N exist (up to 5).
    assert set(it.n_changed for it in items) == set(range(6))


def test_ground_truth_matches_panels(items):
    for it in items:
        assert it.ground_truth == compute_ground_truth(it.panels, it.queried)


def test_every_distractor_changes(items):
    for it in items:
        truth = compute_ground_truth(it.panels, it.distractors)
        assert all(truth.values())


def test_one_object_changes_per_attribute_per_transition(items):
    for it in items:
        for p, q in zip(it.panels[:-1], it.panels[1:]):
            for a in it.queried + it.distractors:
                assert sum(p[j][a] != q[j][a] for j in range(len(p))) <= 1


def test_no_identical_objects_within_a_panel(items):
    for it in items:
        for panel in it.panels:
            keys = [tuple(sorted(o.items())) for o in panel]
            assert len(set(keys)) == len(keys), it.item_id


@pytest.mark.parametrize("n_objects", range(1, 7))
@pytest.mark.parametrize("n_attrs,n_distractors", [(1, 1), (2, 0), (3, 2), (5, 0)])
def test_distinct_objects_across_knobs(n_objects, n_attrs, n_distractors):
    for n_changed in range(n_attrs + 1):
        knobs = Knobs(n_objects, n_attrs, n_changed, 3, n_distractors)
        for seed in range(15):
            it = generate_item(knobs, np.random.default_rng(seed))
            assert it.n_changed == n_changed
            for panel in it.panels:
                assert len({tuple(sorted(o.items())) for o in panel}) == n_objects


def test_text_encodes_ground_truth(items):
    """Decode value multisets from the text alone; a change shows as a multiset difference."""
    for it in items:
        attrs = it.queried + it.distractors
        decoded = [decode_panel(t, attrs) for t in it.panel_texts]
        for panel, dec in zip(it.panels, decoded):
            for a in attrs:
                assert dec[a] == Counter(o[a] for o in panel), (it.item_id, a)
        for a in it.queried:
            text_changed = any(d1[a] != d2[a] for d1, d2 in zip(decoded[:-1], decoded[1:]))
            assert text_changed == it.ground_truth[a], (it.item_id, a)


def test_question_lists_exactly_queried_attributes(items):
    for it in items:
        listed = it.question.split(":", 1)[1].rstrip("?").split(",")
        assert [s.strip() for s in listed] == list(it.queried)
        q_line = [ln for ln in it.prompt.splitlines() if ln.startswith("Question:")]
        assert len(q_line) == 1
        for d in DISTRACTORS:
            assert d not in q_line[0]
        assert set(it.queried) <= set(QUERYABLE)


@pytest.mark.parametrize("seed", range(5))
def test_surface_noise_never_changes_ground_truth(seed):
    """Same seed and knobs, only surface noise varies: identical panels, different wording."""
    base = dict(n_objects=4, n_attrs=4, n_changed=2, n_distractors=1)
    out = [generate_item(Knobs(surface_noise=s, **base), np.random.default_rng(seed))
           for s in range(4)]
    for it in out[1:]:
        assert it.panels == out[0].panels
        assert it.ground_truth == out[0].ground_truth
    assert len({tuple(it.panel_texts) for it in out}) > 1


@pytest.mark.parametrize("seed", range(5))
def test_distractors_never_change_ground_truth(seed):
    """Adding distractors leaves the queried attributes' truth defined by queried values only."""
    for nd in range(3):
        it = generate_item(Knobs(4, 3, 1, 0, nd), np.random.default_rng(seed))
        assert it.n_changed == 1
        assert set(it.ground_truth) == set(it.queried)


@pytest.mark.parametrize("bad", [
    dict(n_objects=0, n_attrs=2, n_changed=0, surface_noise=0, n_distractors=0),
    dict(n_objects=2, n_attrs=6, n_changed=0, surface_noise=0, n_distractors=0),
    dict(n_objects=2, n_attrs=2, n_changed=3, surface_noise=0, n_distractors=0),
    dict(n_objects=2, n_attrs=2, n_changed=0, surface_noise=4, n_distractors=0),
    dict(n_objects=2, n_attrs=2, n_changed=0, surface_noise=0, n_distractors=3),
    dict(n_objects=2, n_attrs=2, n_changed=0, surface_noise=0, n_distractors=0, n_panels=1),
    dict(n_objects=3, n_attrs=1, n_changed=0, surface_noise=0, n_distractors=0),
])
def test_invalid_knobs_rejected(bad):
    with pytest.raises(ValueError):
        generate_item(Knobs(**bad), np.random.default_rng(0))
