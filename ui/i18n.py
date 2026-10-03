"""Тексты окна на русском и английском."""

TEXT = {
    "title": ("AiForPrinter", "AiForPrinter"),
    "tab_live": ("Сейчас", "Now"),
    "tab_alerts": ("Тревоги", "Alerts"),
    "tab_settings": ("Настройки", "Settings"),
    "theme": ("Тема", "Theme"),
    "lang": ("Язык", "Language"),
    "st_printing": ("Печать идёт - смотрю", "Printing - watching"),
    "st_idle": ("Печать не идёт", "Not printing"),
    "st_paused": ("Печать на паузе", "Print paused"),
    "st_preparing": ("Подготовка (нагрев, калибровка)", "Preparing (heating, calibration)"),
    "st_offline": ("camera_server не отвечает", "camera_server does not answer"),
    "st_none": ("Служба не запущена", "Service is not running"),
    "st_unknown": ("Статус принтера непонятен", "Printer status unknown"),
    "heatmap": ("Тепловая карта", "Heat map"),
    "check_now": ("Снимок сейчас", "Check now"),
    "checking": ("Снимаю…", "Taking a shot…"),
    "no_frame": ("Кадра пока нет", "No frame yet"),
    "confidence": ("Уверенность по видам", "Confidence per type"),
    "chart": ("Брак за печать (максимум по видам)", "Failure score over the print (max over types)"),
    "threshold_line": ("порог", "threshold"),
    "no_alerts": ("Тревог пока не было.", "No alerts yet."),
    "answer_ok": ("отмечено: всё нормально", "marked: all good"),
    "answer_pause": ("поставлена пауза", "paused"),
    "answer_none": ("без ответа", "no answer"),
    "interval": ("Снимок раз в, с", "Shot every, s"),
    "threshold": ("Порог уверенности", "Confidence threshold"),
    "window": ("Решение по кадрам", "Decide over frames"),
    "need": ("Из них с браком", "Of them failing"),
    "cooldown": ("Тишина после тревоги, мин", "Quiet after an alert, min"),
    "telegram": ("Тревоги в Telegram (бот n8n)", "Alerts to Telegram (n8n bot)"),
    "autopause": ("Автопауза без человека", "Auto-pause without a human"),
    "autopause_hint": ("Выключена по умолчанию. Включать только после недели наблюдений, когда ложных тревог мало. "
                       "Остановку печати программа не делает никогда.",
                       "Off by default. Turn on only after a week of observation with few false alarms. "
                       "The program never stops a print."),
    "autopause_confirm": ("Автопауза поставит печать на паузу без вас, по решению сети. Включить?",
                          "Auto-pause will pause the print without you, on the network's decision. Turn it on?"),
    "camera_url": ("Адрес camera_server", "camera_server address"),
    "save": ("Сохранить", "Save"),
    "saved": ("Сохранено. Служба подхватит на следующем снимке.", "Saved. The service picks it up on the next shot."),
    "bad_url": ("Адрес: только http://хост:порт", "Address: http://host:port only"),
    "passport": ("Паспорт модели", "Model passport"),
    "passport_none": ("Сеть ещё не обучена (нет model/passport.json).", "No trained network yet (model/passport.json missing)."),
    "fa_per_hour": ("ложных тревог в час нормальной печати", "false alarms per hour of normal printing"),
    "model_missing": ("Нет обученной сети", "No trained network"),
}


def tr(key, lang="ru"):
    pair = TEXT.get(key)
    if pair is None:
        return key
    return pair[1] if lang == "en" else pair[0]
