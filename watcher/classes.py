"""Виды кадров, которые различает сеть. Порядок = порядок выходов сети, менять только вместе с обучением."""

CLASSES = ("normal", "spaghetti", "detached", "blob", "no_extrusion")
FAILURES = CLASSES[1:]

NAMES = {
    "ru": {
        "normal": "норма",
        "spaghetti": "спагетти",
        "detached": "отрыв от стола",
        "blob": "ком на сопле",
        "no_extrusion": "нет пластика",
    },
    "en": {
        "normal": "normal",
        "spaghetti": "spaghetti",
        "detached": "detached from bed",
        "blob": "blob on nozzle",
        "no_extrusion": "no extrusion",
    },
}


def name(cls, lang="ru"):
    return NAMES.get(lang, NAMES["ru"]).get(cls, cls)
