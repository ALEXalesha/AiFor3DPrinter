"""Пять тем Paint Pro («Вид → Тема»): Стеклянная, Строгая, Светлая, Ночная, Тёплая.

Значения - токены Paint (Paint/paint-pro-electron/paint-pro.html, как в калькуляторах):
основа окна (--app-base), цвет и прозрачность стекла (--ink и --glass-bg*), текст,
акцент, «опасно» и «хорошо». Qt не умеет размытие под стеклом, поэтому стекло здесь -
то же стекло, наложенное на основу: цвет считается смешением, а не рисуется альфой.
"""

TOKENS = {
    # основа, ink (R G B), стекло: обычное / сильное / рамка, текст, акцент, второй акцент
    "glass": ("#12101f", (255, 255, 255), 0.08, 0.14, 0.18, "#f4f4f8", "#5b8def", "#9d5bef"),
    "formal": ("#1f242e", (255, 255, 255), 0.055, 0.10, 0.11, "#e6e8ec", "#4a7ab8", "#5c7fa6"),
    "light": ("#e8ecf3", (22, 26, 38), 0.045, 0.10, 0.14, "#1b2030", "#3a6fd8", "#7a52d0"),
    "night": ("#08090e", (255, 255, 255), 0.045, 0.085, 0.09, "#dcdde3", "#4668a8", "#6a5aa0"),
    "warm": ("#1d1610", (255, 236, 210), 0.07, 0.13, 0.16, "#f5ead9", "#d08a3c", "#b4562f"),
}
NAMES = {
    "ru": {"glass": "Стеклянная", "formal": "Строгая", "light": "Светлая", "night": "Ночная", "warm": "Тёплая"},
    "en": {"glass": "Glass", "formal": "Formal", "light": "Light", "night": "Night", "warm": "Warm"},
}
DANGER = "#ff5b5b"
SUCCESS = "#2dd4a8"


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _hex(c):
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v))) for v in c)


def mix(a, b, t):
    """Цвет a, поверх которого лежит b с непрозрачностью t."""
    a, b = (_rgb(a) if isinstance(a, str) else a), (_rgb(b) if isinstance(b, str) else b)
    return _hex(tuple(x * (1 - t) + y * t for x, y in zip(a, b)))


def palette(name):
    base, ink, g1, g2, gb, text, accent, accent2 = TOKENS.get(name, TOKENS["glass"])
    light = name == "light"
    return {
        "bg": base,
        "surface": mix(base, (255, 255, 255), 0.65) if light else mix(base, ink, g1),
        "surface2": mix(base, ink, g2),
        "border": mix(base, ink, gb),
        "text": text,
        "muted": mix(base, text, 0.6),
        "accent": accent,
        "accent2": accent2,
        "accent_text": "#ffffff",
        "danger": DANGER,
        "good": SUCCESS,
        "grid": mix(base, ink, g1 * 1.5),
    }


def qss(name):
    p = palette(name)
    return f"""
* {{ font-family: "Segoe UI", "Noto Sans", sans-serif; font-size: 14px; color: {p['text']}; }}
QMainWindow, QWidget#page, QDialog {{ background: {p['bg']}; }}
QLabel, QCheckBox {{ background: transparent; }}
QLabel#title {{ font-size: 22px; font-weight: 700; }}
QLabel#muted {{ color: {p['muted']}; }}
QLabel#big {{ font-size: 26px; font-weight: 700; }}
QLabel#stateOk {{ color: {p['good']}; font-weight: 600; }}
QLabel#stateBad {{ color: {p['danger']}; font-weight: 600; }}
QLabel#stateIdle {{ color: {p['muted']}; font-weight: 600; }}
QFrame#card {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 14px; }}
QPushButton {{ background: {p['surface2']}; border: 1px solid {p['border']}; border-radius: 10px; padding: 8px 16px; }}
QPushButton:hover {{ border-color: {p['accent']}; }}
QPushButton#primary {{ background: {p['accent']}; color: {p['accent_text']}; border-color: {p['accent']}; font-weight: 600; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 8px 16px; color: {p['muted']}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {p['text']}; border-bottom: 2px solid {p['accent']}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {p['surface2']}; border: 1px solid {p['border']}; border-radius: 8px; padding: 5px 9px; min-height: 22px;
}}
QComboBox {{ padding-right: 26px; min-width: 120px; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {p['accent']}; }}
QComboBox::drop-down {{ border: none; width: 24px; background: transparent; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; }}
QComboBox QAbstractItemView {{ background: {p['surface']}; border: 1px solid {p['border']};
    selection-background-color: {p['accent']}; selection-color: {p['accent_text']}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}
QCheckBox {{ spacing: 9px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid {p['border']}; background: {p['surface2']}; }}
QCheckBox::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; }}
QProgressBar {{ background: {p['surface2']}; border: none; border-radius: 6px; height: 12px; text-align: right; }}
QProgressBar::chunk {{ background: {p['accent']}; border-radius: 6px; }}
QProgressBar#fail::chunk {{ background: {p['danger']}; border-radius: 6px; }}
QListWidget {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 12px; padding: 4px; }}
QListWidget::item {{ padding: 6px; border-radius: 8px; }}
QListWidget::item:selected {{ background: {p['surface2']}; color: {p['text']}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {p['border']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
