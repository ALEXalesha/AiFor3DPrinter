"""Разбор ответа camera_server на /status.

camera_server отдаёт не JSON, а готовый текст для Telegram (fmt_status), например:

    🖨 cube.gcode
    Статус: печатает
    Слой 50/200 (25%)
    ...

Служба смотрит только строку «Статус: ...». Сеть работает лишь при «печатает»
(PrintInfo.Status 13 у Centauri Carbon): во время нагрева, калибровки и паузы стол
пустой или ничего не происходит - кадры там дают ложные тревоги и ничего не спасают.
"""
import re
from dataclasses import dataclass

# текст fmt_status -> состояние для службы
STATE_BY_TEXT = {
    "печатает": "printing",
    "пауза": "paused",
    "ставится на паузу": "paused",
    "нагрев": "preparing",
    "калибровка стола": "preparing",
    "парковка осей": "preparing",
    "проверка файла": "preparing",
    "простаивает": "idle",
    "завершено": "idle",
    "остановлено": "idle",
    "останавливается": "idle",
}


@dataclass(frozen=True)
class PrinterStatus:
    state: str              # printing / paused / preparing / idle / unknown
    text: str = ""          # строка статуса как есть
    layer: int | None = None
    total_layers: int | None = None
    filename: str = ""

    @property
    def printing(self):
        return self.state == "printing"


_STATUS_RE = re.compile(r"^\s*Статус:\s*(.+?)\s*$", re.M)
_LAYER_RE = re.compile(r"Слой\s+(\d+)\s*/\s*(\d+)")
_FILE_RE = re.compile(r"^\s*🖨\s*(.+?)\s*$", re.M)


def parse_status(text):
    """Текст /status -> PrinterStatus. Незнакомое - unknown (служба такое не смотрит)."""
    if not isinstance(text, str):
        return PrinterStatus("unknown")
    m = _STATUS_RE.search(text)
    if not m:
        return PrinterStatus("unknown", text[:200])
    word = m.group(1).strip().lower()
    state = STATE_BY_TEXT.get(word, "unknown")
    layer = total = None
    lm = _LAYER_RE.search(text)
    if lm:
        layer, total = int(lm.group(1)), int(lm.group(2))
    fm = _FILE_RE.search(text)
    return PrinterStatus(state, word, layer, total, fm.group(1) if fm else "")
