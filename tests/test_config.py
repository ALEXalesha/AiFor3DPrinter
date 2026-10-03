"""Настройки: мусор не ломает, автопауза включается только явным true."""
import json

from hypothesis import given
from hypothesis import strategies as st

from watcher import config


def test_defaults_autopause_off():
    assert config.DEFAULTS["auto_pause"] is False
    assert config.load()["auto_pause"] is False


def test_autopause_only_exact_true():
    for v in ("true", 1, "1", "yes", [True], {"x": 1}, None, 1.0):
        assert config.normalize({"auto_pause": v})["auto_pause"] is False, v
    assert config.normalize({"auto_pause": True})["auto_pause"] is True


def test_limits_clamped():
    c = config.normalize({"interval_s": 0.1, "threshold": 5, "window": 3, "need": 9, "cooldown_s": -5})
    assert c["interval_s"] == 5 and c["threshold"] == 0.99 and c["window"] == 3 and c["need"] == 3
    assert c["cooldown_s"] == 0


def test_corrupt_file_gives_defaults(_own_data_dir):
    p = config.config_path()
    open(p, "w", encoding="utf-8").write("{oops")
    assert config.load() == config.DEFAULTS


def test_save_roundtrip(_own_data_dir):
    c = dict(config.DEFAULTS, interval_s=20, theme="warm", lang="en")
    assert config.save(c)
    assert config.load() == config.normalize(c)
    assert json.load(open(config.config_path(), encoding="utf-8"))["interval_s"] == 20


def test_bot_dir_default_under_n8n_pc(monkeypatch, tmp_path):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert config.bot_dir(config.DEFAULTS).endswith("n8n_pc\\run\\aifp") or \
        config.bot_dir(config.DEFAULTS).endswith("n8n_pc/run/aifp")


anything = st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True) | st.text(max_size=20),
                        lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=8), c, max_size=3),
                        max_leaves=8)


@given(st.dictionaries(st.sampled_from(list(config.DEFAULTS) + ["x"]), anything) | anything)
def test_law_normalize_total_and_safe(raw):
    c = config.normalize(raw)
    assert set(c) == set(config.DEFAULTS)
    for k, (lo, hi) in config.LIMITS.items():
        assert lo <= c[k] <= hi
    assert 1 <= c["need"] <= c["window"]
    assert c["auto_pause"] is (isinstance(raw, dict) and raw.get("auto_pause") is True)
    assert c["theme"] in config.THEMES and c["lang"] in config.LANGS
    assert config.normalize(c) == c
