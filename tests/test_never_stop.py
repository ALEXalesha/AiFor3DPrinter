"""Главный закон безопасности: служба НИКОГДА не останавливает печать.

Остановить печать (/stopprint) может только человек. Служба смотрит, сообщает и -
только при явно включённой автопаузе - ставит паузу. Законы проверяют это на всём
пространстве входов, а tools/mutate.py убеждается, что каждый закон падает, если
правило в коде сломать.
"""
import ast
import os

from fake_camera import STATUS_IDLE, STATUS_PRINTING
from helpers import Clock, FakeModel, bad, cfg_loader, quiet_log
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from watcher.camera_client import ALLOWED, CameraClient, ForbiddenEndpoint, check_base_url
from watcher.classes import CLASSES
from watcher.service import Service

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def route(url):
    """Как camera_server разберёт путь запроса: self.path.split('?')[0].strip('/')."""
    rest = url.split("://", 1)[1]
    path = "/" + rest.split("/", 1)[1] if "/" in rest else "/"
    return path.split("?")[0].strip("/")


def test_allowed_set_is_exactly_three_safe_paths():
    assert ALLOWED == frozenset({"snap", "status", "pause"})
    assert not hasattr(CameraClient, "stop") and not hasattr(CameraClient, "stopprint")


@given(st.one_of(st.sampled_from(["stopprint", "stopprint/", "/stopprint", "stopprint?", "resume", "print",
                                  "snap/../stopprint", "STOPPRINT", "pause/../stopprint"]),
                 st.text(max_size=30)))
def test_law_client_never_builds_a_stop_url(endpoint):
    c = CameraClient("http://127.0.0.1:8099")
    try:
        url = c.url_for(endpoint)
    except ForbiddenEndpoint:
        return
    assert route(url) in ALLOWED


url_parts = st.lists(st.sampled_from(["http://", "https://", "127.0.0.1", ":8099", "/", "stopprint", "?", "x=",
                                      "#", "@", "u:p", "localhost", "/snap", "%2F", ":", "[::1]", " "]),
                     max_size=8).map("".join)


@given(st.one_of(url_parts, st.text(max_size=40)), st.sampled_from(sorted(ALLOWED)))
def test_law_any_base_url_routes_only_to_allowed(base, endpoint):
    try:
        check_base_url(base)
        c = CameraClient(base)
    except ValueError:
        return
    assert route(c.url_for(endpoint)) == endpoint


statuses = st.sampled_from([STATUS_PRINTING, STATUS_IDLE, "Статус: пауза", "Статус: нагрев", "Статус: код 7",
                            "мусор", STATUS_PRINTING.replace("cube", "b")])
modes = st.sampled_from(["ok", "ok", "ok", "503", "garbage", "drop"])
probs = st.one_of(st.sampled_from([bad(c, 0.95) for c in CLASSES[1:]] + [{"normal": 1.0}]),
                  st.dictionaries(st.sampled_from(CLASSES), st.floats(0, 1), max_size=5))


@settings(max_examples=120, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.tuples(statuses, modes, probs), min_size=1, max_size=12), st.booleans(),
       st.floats(0.3, 0.99), st.integers(1, 10), st.integers(1, 10), st.booleans())
def test_law_service_never_stops_print(fake_camera, steps, auto_pause, thr, window, need, telegram):
    fake_camera.raw_paths.clear()
    fake_camera.routes.clear()
    load = cfg_loader(camera_url=fake_camera.url, auto_pause=auto_pause, threshold=thr, window=window, need=need,
                      cooldown_s=0, telegram=telegram)
    model = FakeModel(seq=[p for _, _, p in steps])
    svc = Service(model, client=CameraClient(fake_camera.url, {"snap": 2, "status": 2, "pause": 2}),
                  cfg_loader=load, clock=Clock(), log=quiet_log())
    for status, mode, _ in steps:
        fake_camera.status_text, fake_camera.mode = status, mode
        svc.step()
    assert "stopprint" not in fake_camera.routes
    assert set(fake_camera.routes) <= ALLOWED
    if not auto_pause:
        assert "pause" not in fake_camera.routes


def test_default_config_never_pauses_even_on_endless_failures(fake_camera):
    load = cfg_loader(camera_url=fake_camera.url, cooldown_s=0)
    svc = Service(FakeModel(default=bad()), client=CameraClient(fake_camera.url), cfg_loader=load,
                  clock=Clock(), log=quiet_log())
    for _ in range(30):
        svc.step()
    assert fake_camera.routes.count("snap") == 30
    assert "pause" not in fake_camera.routes and "stopprint" not in fake_camera.routes


def _code_strings_and_names(path):
    """Строки и имена в коде, кроме докстрингов и комментариев."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docs.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            yield node.value
        elif isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr


def test_no_product_code_mentions_stopprint():
    """Ни в службе, ни в окне, ни в инструментах нет строки или имени со stopprint."""
    bad_hits = []
    for top in ("watcher", "ui", "model", "dataset", "tools", "main.py"):
        base = os.path.join(ROOT, top)
        files = [base] if base.endswith(".py") else [
            os.path.join(d, f) for d, _, fs in os.walk(base) for f in fs if f.endswith(".py")]
        for f in files:
            if not os.path.exists(f) or os.path.basename(f) == "mutate.py":
                continue
            for s in _code_strings_and_names(f):
                if "stopprint" in s.lower().replace("_", "").replace("-", ""):
                    bad_hits.append((os.path.relpath(f, ROOT), s[:60]))
    assert bad_hits == []


def test_no_product_code_opens_urls_except_camera_client():
    """Сетевой запрос к camera_server делает только camera_client (urlopen больше нигде нет)."""
    hits = []
    for top in ("watcher", "ui"):
        for d, _, fs in os.walk(os.path.join(ROOT, top)):
            for f in fs:
                if f.endswith(".py") and f != "camera_client.py":
                    text = open(os.path.join(d, f), encoding="utf-8").read()
                    if "urlopen" in text or "http.client" in text or "requests" in text:
                        hits.append(f)
    assert hits == []
