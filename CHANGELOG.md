# Changelog

No releases yet. Beta: the service and the Telegram alerts work; a week of observation on real prints and the first release are ahead.

## Unreleased

- 2026-10-17 - A Telegram alert brings two photos: a close-up of the failure and the whole frame.
- 2026-10-17 - Published on GitHub.
- 2026-10-04 - Every snapshot goes to the log and to local storage, `main.py review` shows the scores of past prints; the first 10 layers are checked every 5 seconds and more sensitively.
- 2026-10-04 - The window: labels and fields with proper spacing in every theme and at the minimum size.
- 2026-10-03 - Trained network (MobileNetV3 with a "where is the failure" heat map, ONNX, about 5 ms per frame on the CPU): 0.41 false alerts per hour; several parts on the bed are no longer taken for spaghetti; "All good" silences that kind of failure until the end of the print (0.11 alerts per hour).
- 2026-10-03 - PySide6 window: a live snapshot with the heat map, confidence per failure kind, a chart for the print, alert history with answers, settings, five themes, RU/EN; auto-pause turns on only with confirmation.
- 2026-10-03 - The service: watches only while the printer is printing, alerts on 3 of 4 frames with a 10-minute pause between alerts, sends alerts to the Telegram bot and never stops a print.

---

Выпусков пока нет. Бета: служба и тревоги в Telegram работают; впереди неделя наблюдения на настоящих печатях и первый выпуск.

## Не выпущено

- 2026-10-17 - Тревога в Telegram приходит двумя фото: крупно место брака и весь кадр.
- 2026-10-17 - Опубликован на GitHub.
- 2026-10-04 - Каждый снимок пишется в журнал и в локальное хранилище, `main.py review` показывает оценки прошлых печатей; первые 10 слоёв проверяются раз в 5 секунд и чувствительнее.
- 2026-10-04 - Окно: подписи и поля с нормальными отступами во всех темах и в минимальном размере.
- 2026-10-03 - Обученная сеть (MobileNetV3 с тепловой картой «где брак», ONNX, около 5 мс на кадр на процессоре): 0.41 ложной тревоги в час; несколько деталей на столе больше не принимаются за спагетти; «Всё нормально» глушит этот вид брака до конца печати (0.11 тревоги в час).
- 2026-10-03 - Окно на PySide6: живой снимок с тепловой картой, уверенность по видам брака, график за печать, история тревог с ответами, настройки, пять тем, RU/EN; автопауза включается только с подтверждением.
- 2026-10-03 - Служба: смотрит только во время печати, тревога по 3 из 4 кадров с паузой 10 минут между тревогами, тревоги уходят боту в Telegram, печать никогда не останавливается.
