import numpy as np
import pytest
from scipy.stats import norm

from changedet.generator import generate_set
from changedet.responders import Response
from changedet.scoring import build_trials, sdt, summarize, summarize_by_level


def test_sdt_hand_computed():
    out = sdt(hits=8, n_signal=10, fas=2, n_noise=10)
    H, F = 8.5 / 11, 2.5 / 11
    assert out["H"] == pytest.approx(H) and out["F"] == pytest.approx(F)
    assert out["dprime"] == pytest.approx(norm.ppf(H) - norm.ppf(F))
    assert out["c"] == pytest.approx(-(norm.ppf(H) + norm.ppf(F)) / 2)


def test_sdt_extremes_finite():
    out = sdt(hits=10, n_signal=10, fas=0, n_noise=10)
    assert np.isfinite(out["dprime"]) and np.isfinite(out["c"])
    assert out["H"] == pytest.approx(10.5 / 11) and out["F"] == pytest.approx(0.5 / 11)
    assert out["c"] == pytest.approx(0.0)  # symmetric extremes -> unbiased


def test_sdt_liberal_bias_negative_c():
    assert sdt(9, 10, 6, 10)["c"] < 0
    assert sdt(5, 10, 1, 10)["c"] > 0


def _answer(it, attrs):
    return Response(it.item_id, "CHANGED: " + (", ".join(attrs) or "none"), 2.0)


def test_perfect_responder():
    items = generate_set(4, 50, 1)
    resp = [_answer(it, [a for a in it.queried if it.ground_truth[a]]) for it in items]
    s = summarize(build_trials(items, resp))
    assert s["accuracy"] == 1.0 and s["exact_accuracy"] == 1.0
    assert s["hit_rate_raw"] == 1.0 and s["fa_rate_raw"] == 0.0
    assert s["mean_rt"] == 2.0


def test_report_everything_responder():
    items = generate_set(3, 50, 1)
    resp = [_answer(it, it.queried) for it in items]
    s = summarize(build_trials(items, resp))
    assert s["hit_rate_raw"] == 1.0 and s["fa_rate_raw"] == 1.0
    assert s["dprime"] == pytest.approx(
        norm.ppf((s["n_signal"] + .5) / (s["n_signal"] + 1))
        - norm.ppf((s["n_noise"] + .5) / (s["n_noise"] + 1)))


def test_unparseable_counts_as_no_report_and_is_counted():
    items = generate_set(2, 20, 3)
    resp = [Response(it.item_id, "I am not sure.", 1.0) for it in items]
    t = build_trials(items, resp)
    assert not t.reported.any()
    assert summarize(t)["unparseable_rate"] == 1.0


def test_distractor_reports_ignored_but_counted():
    items = [it for it in generate_set(4, 20, 3)]
    resp = [_answer(it, list(it.distractors)) for it in items]
    t = build_trials(items, resp)
    assert not t.reported.any()
    assert summarize(t)["extraneous_reports"] == sum(len(it.distractors) for it in items)


def test_mismatched_ids_rejected():
    items = generate_set(1, 3, 0)
    with pytest.raises(ValueError):
        build_trials(items, [Response("x", "CHANGED: none", 1.0)] * 3)


def test_summary_by_level_partitions(items):
    resp = [_answer(it, []) for it in items]
    t = build_trials(items, resp)
    by = summarize_by_level(t)
    assert sorted(by) == [1, 2, 3, 4]
    assert sum(v["n_items"] for v in by.values()) == len(items)
    assert sum(v["n_signal"] + v["n_noise"] for v in by.values()) == len(t.changed)
