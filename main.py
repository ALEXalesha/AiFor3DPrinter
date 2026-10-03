"""AiForPrinter: нейросеть смотрит в камеру принтера и замечает брак.

    python main.py watch            служба (раз в N секунд, только во время печати)
    python main.py ui               окно: живой снимок, тепловая карта, история, настройки
    python main.py check            один снимок сейчас: оценка сетью (только чтение /snap)
    python main.py selftest FILE    проверка без экрана: сеть грузится, окно строится
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def cmd_check():
    from watcher import config
    from watcher.camera_client import CameraClient
    from watcher.service import load_model
    cfg = config.load()
    c = CameraClient(cfg["camera_url"])
    st = c.status()
    model = load_model()
    out = {"status": st.state, "text": st.text}
    if model is not None:
        pred = model.predict(c.snapshot())
        out["probs"] = {k: round(v, 3) for k, v in pred.probs.items()}
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_selftest(path):
    import offscreen
    offscreen.setup(force=True)
    lines = []
    from watcher.service import load_model
    model = load_model()
    lines.append(f"model: {'ok ' + str(model.size) if model else 'нет'}")
    if model is not None:
        import numpy as np
        pred = model.predict(np.full((360, 640, 3), 60, np.uint8))
        lines.append("predict: " + ", ".join(f"{k}={v:.3f}" for k, v in pred.probs.items()))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from ui.app import MainWindow
    w = MainWindow(remember=False)
    w.refresh()
    lines.append(f"window: {w.width()}x{w.height()} tabs={w.tabs.count()}")
    w.close()
    app.processEvents()
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "ui"
    if cmd == "watch":
        from watcher.service import main as watch
        watch()
    elif cmd == "ui":
        from ui.app import run
        sys.exit(run())
    elif cmd == "check":
        cmd_check()
    elif cmd == "selftest":
        cmd_selftest(argv[2] if len(argv) > 2 else "selftest.txt")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
