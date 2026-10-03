"""Решение по нескольким кадрам: примеры и законы на всём пространстве входов."""
from hypothesis import given
from hypothesis import strategies as st

from watcher.classes import CLASSES, FAILURES
from watcher.decision import Decider, frame_vote

N = {"normal": 0.97, "spaghetti": 0.01, "detached": 0.01, "blob": 0.005, "no_extrusion": 0.005}


def bad(cls="spaghetti", p=0.9):
    d = {c: (1 - p) / 4 for c in CLASSES}
    d[cls] = p
    return d


def test_single_bad_frame_is_not_an_alarm():
    d = Decider(0.7, 4, 3, 600)
    assert d.update(bad(), 0) is None
    for t in range(1, 10):
        assert d.update(N, t * 15) is None


def test_three_of_four_alarm():
    d = Decider(0.7, 4, 3, 600)
    assert d.update(bad(), 0) is None
    assert d.update(N, 15) is None
    assert d.update(bad(), 30) is None
    a = d.update(bad(), 45)
    assert a and a.cls == "spaghetti" and a.votes == 3 and a.window == 4


def test_cooldown_then_alarm_again():
    d = Decider(0.7, 4, 3, 600)
    alarms = [t for t in range(0, 1500, 15) if d.update(bad(), t)]
    assert alarms[0] == 30 and all(b - a >= 600 for a, b in zip(alarms, alarms[1:]))


def test_reset_forgets_votes_and_cooldown():
    d = Decider(0.7, 4, 3, 600)
    for t in (0, 15, 30):
        d.update(bad(), t)
    d.reset()
    assert d.update(bad(), 45) is None and d.update(bad(), 60) is None
    assert d.update(bad(), 75) is not None


def test_old_votes_expire():
    d = Decider(0.7, 4, 3, 0, max_age_s=100)
    d.update(bad(), 0)
    d.update(bad(), 10)
    assert d.update(bad(), 500) is None


def test_vote_is_majority_class():
    d = Decider(0.7, 5, 3, 0)
    for t, c in enumerate(["blob", "detached", "blob", "detached", "blob"]):
        a = d.update(bad(c), t)
    assert a.cls == "blob"


probs_st = st.dictionaries(st.sampled_from(CLASSES), st.floats(0, 1), min_size=0, max_size=5)


@given(st.lists(probs_st, max_size=40), st.floats(0.3, 0.99), st.integers(1, 10), st.integers(1, 10),
       st.integers(0, 2000))
def test_law_alarm_needs_enough_votes_in_window(seq, thr, window, need, cooldown):
    d = Decider(thr, window, need, cooldown)
    need = min(need, window)
    hist = []
    last = None
    for i, p in enumerate(seq):
        t = i * 15
        hist.append(frame_vote(p, thr))
        a = d.update(p, t)
        if a:
            recent = hist[-window:]
            assert sum(v is not None for v in recent) >= need
            assert a.cls in FAILURES and a.cls in recent
            assert last is None or t - last >= cooldown
            last = t


@given(st.lists(st.floats(0, 0.6999), max_size=60), st.integers(1, 10), st.integers(1, 10))
def test_law_below_threshold_never_alarms(ps, window, need):
    d = Decider(0.7, window, need, 0)
    for i, p in enumerate(ps):
        assert d.update({"normal": 1 - p, "spaghetti": p}, i) is None


@given(st.lists(st.just(0), min_size=0, max_size=60))
def test_law_all_normal_never_alarms(seq):
    d = Decider(0.3, 1, 1, 0)
    for i, _ in enumerate(seq):
        assert d.update({"normal": 1.0}, i) is None
