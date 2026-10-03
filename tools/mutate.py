"""Мутационная проверка: ломаем правило в копии кода и смотрим, что нужная проверка падает.

    python tools/mutate.py           все мутации
    python tools/mutate.py 3 5       только мутации 3 и 5

Зелёный набор тестов сам по себе ничего не доказывает: проверка должна падать, если
правило сломать. Каждая мутация - (файл, было, стало, какие тесты должны упасть).
Копия кода - во временной папке, рабочая папка не трогается. Итог: KILLED / SURVIVED.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = "stop" + "print"     # слово целиком в коде продукта запрещено законом - собираем его здесь

MUTATIONS = [
    ("клиент разрешает остановку печати", "watcher/camera_client.py",
     'ALLOWED = frozenset({"snap", "status", "pause"})', f'ALLOWED = frozenset({{"snap", "status", "pause", "{STOP}"}})',
     "tests/test_never_stop.py::test_allowed_set_is_exactly_three_safe_paths "
     "tests/test_never_stop.py::test_law_client_never_builds_a_stop_url"),
    # Адрес защищён двумя слоями: проверка отказывает, а принятый адрес пересобирается из
    # хоста и порта (путь и запрос отбрасываются). Каждый слой ловит своя проверка.
    ("адрес camera_server с путём и ?запросом проходит", "watcher/camera_client.py",
     'if parts.path not in ("", "/") or parts.query or parts.fragment or "?" in url or "#" in url:',
     'if parts.fragment or "#" in url:',
     "tests/test_camera_client.py"),
    ("принятый адрес не пересобирается из хоста и порта", "watcher/camera_client.py",
     'return f"http://{host}:{port or 80}"', 'return url.strip().rstrip("/")',
     "tests/test_camera_client.py"),
    ("служба вместо паузы останавливает печать", "watcher/service.py",
     "                ok = self.client.pause()",
     "                ok = __import__('urllib.request').request.urlopen(self.client.base + '/" + STOP + "', timeout=3).status == 200",
     "tests/test_never_stop.py -k law_service"),
    ("автопауза включена по умолчанию", "watcher/config.py",
     '"auto_pause": False,       # пауза без человека', '"auto_pause": True,       # пауза без человека',
     "tests/test_config.py tests/test_never_stop.py"),
    ("автопауза включается строкой \"true\"", "watcher/config.py",
     'cfg["auto_pause"] = raw.get("auto_pause") is True', 'cfg["auto_pause"] = bool(raw.get("auto_pause"))',
     "tests/test_config.py"),
    ("служба зовёт паузу без проверки настройки", "watcher/service.py",
     'if self.cfg["auto_pause"] is True:', "if True:",
     "tests/test_never_stop.py tests/test_service.py"),
    ("тревога по одному кадру", "watcher/decision.py",
     "if len(votes) < self.need:", "if len(votes) < 1:",
     "tests/test_decision.py"),
    ("нет паузы между тревогами", "watcher/decision.py",
     "if self.last_alarm_t is not None and t - self.last_alarm_t < self.cooldown_s:", "if False:",
     "tests/test_decision.py"),
    ("служба смотрит и во время нагрева", "watcher/service.py",
     "        if not status.printing:", '        if status.state in ("idle",):',
     "tests/test_service.py tests/test_never_stop.py"),
    ("сбой camera_server роняет службу", "watcher/service.py",
     "        except CameraError as e:\n            self.offline_streak += 1",
     "        except ZeroDivisionError as e:\n            self.offline_streak += 1",
     "tests/test_service.py"),
    ("бот получает тревогу при выключенном Telegram", "watcher/alerts.py",
     '    if cfg.get("telegram"):', "    if True:",
     "tests/test_service.py"),
    ("ответ бота с чужим именем файла принимается", "watcher/alerts.py",
     'if not ID_RE.match(aid) or kind not in ("ok", "pause"):', 'if kind not in ("ok", "pause"):',
     "tests/test_service.py"),
    ("разбиение по кадру, а не по печати", "dataset/common.py",
     'h = int(hashlib.sha1((salt + "|" + group).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF',
     "import random; h = random.random()",
     "tests/test_dataset.py"),
    ("«Всё нормально» глушит и следующие печати", "watcher/decision.py",
     "        self.last_alarm_t = None\n        self.muted = set()\n\n    def mute(",
     "        self.last_alarm_t = None\n\n    def mute(",
     "tests/test_decision.py tests/test_service.py"),
    ("кнопка «Пауза» тоже глушит тревоги", "watcher/service.py",
     'if kind != "ok":\n            return', 'if kind not in ("ok", "pause"):\n            return',
     "tests/test_service.py"),
    ("окно включает автопаузу без подтверждения", "ui/app.py",
     "if on and not self.confirm(tr(\"autopause_confirm\", self.lang)):", "if False:",
     "tests/test_ui.py"),
]


def copy_tree(dst):
    def ignore(d, names):
        return [n for n in names if n in (".venv", ".git", "__pycache__", ".pytest_cache", ".hypothesis", "data",
                                          "build", "dist", ".ci-venv")]
    shutil.copytree(ROOT, dst, ignore=ignore)


def run(sel=None):
    py = sys.executable
    results = []
    for i, (title, path, old, new, tests) in enumerate(MUTATIONS, 1):
        if sel and i not in sel:
            continue
        tmp = tempfile.mkdtemp(prefix="aifp-mut-")
        work = os.path.join(tmp, "repo")
        try:
            copy_tree(work)
            f = os.path.join(work, path)
            src = open(f, encoding="utf-8").read()
            if old not in src:
                results.append((i, title, "NOT FOUND", ""))
                continue
            open(f, "w", encoding="utf-8", newline="\n").write(src.replace(old, new, 1))
            env = dict(os.environ, HYPOTHESIS_PROFILE="quick", PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([py, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests.split()],
                               cwd=work, env=env, capture_output=True, text=True, timeout=900)
            failed = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
            if r.returncode in (2, 3, 4, 5):          # проверки не собрались / не нашлись - это не «поймано»
                results.append((i, title, "ERROR", r.stdout.strip().splitlines()[-1][:100] if r.stdout.strip() else ""))
                print(f"{i:2}. ERROR     {title}  <- {results[-1][3]}", flush=True)
                continue
            results.append((i, title, "KILLED" if r.returncode else "SURVIVED",
                            failed[0][7:110] if failed else r.stdout.strip().splitlines()[-1][:100]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)      # своя временная копия кода
        print(f"{results[-1][0]:2}. {results[-1][2]:9} {results[-1][1]}  <- {results[-1][3]}", flush=True)
    survived = [r for r in results if r[2] != "KILLED"]
    print(f"\n{len(results) - len(survived)} из {len(results)} мутаций пойманы")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(run({int(a) for a in sys.argv[1:]} or None))
