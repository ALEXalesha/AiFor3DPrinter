"""Клиент camera_server (n8n_pc/scripts/camera_server.py, порт 8099).

Клиент умеет ровно три запроса: снимок (/snap), статус (/status) и паузу (/pause).
Остановки печати в нём нет и быть не может:

- набор разрешённых путей ALLOWED - замороженный, другого пути _get не отправит;
- адрес сервера проверяется: только http://хост:порт без пути, запроса и якоря.
  Иначе адрес вида http://127.0.0.1:8099/stopprint?x= превратил бы /snap в
  /stopprint?x=/snap, а camera_server режет всё после «?» и остановил бы печать.

Это проверяют законы в tests/test_never_stop.py, а мутационная проверка
(tools/mutate.py) убеждается, что законы падают, если правило сломать.
"""
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from watcher.status import PrinterStatus, parse_status

ALLOWED = frozenset({"snap", "status", "pause"})

SNAP_TIMEOUT = 25      # camera_server собирает кадр из MJPEG до 10 с плюс два WebSocket-запроса
STATUS_TIMEOUT = 20
PAUSE_TIMEOUT = 30
MAX_JPEG = 20 * 1024 * 1024


class CameraError(Exception):
    """camera_server не ответил или ответил не тем. Служба ждёт и пишет в журнал."""


class ForbiddenEndpoint(Exception):
    """Попытка отправить путь вне ALLOWED - ошибка программы, не сети."""


def check_base_url(url):
    """http://хост[:порт][/] -> 'http://хост:порт'. Всё остальное - ValueError."""
    if not isinstance(url, str):
        raise ValueError("адрес camera_server должен быть строкой")
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as e:
        raise ValueError(f"плохой адрес camera_server: {e}") from None
    if parts.scheme != "http" or not parts.hostname:
        raise ValueError("адрес camera_server: только http://хост:порт")
    if parts.path not in ("", "/") or parts.query or parts.fragment or "?" in url or "#" in url:
        raise ValueError("адрес camera_server: без пути, ?запроса и #якоря")
    if parts.username or parts.password or "@" in parts.netloc:
        raise ValueError("адрес camera_server: без логина в адресе")
    host = parts.hostname
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port or 80}"


class CameraClient:
    def __init__(self, base_url="http://127.0.0.1:8099", timeouts=None):
        self.base = check_base_url(base_url)
        t = {"snap": SNAP_TIMEOUT, "status": STATUS_TIMEOUT, "pause": PAUSE_TIMEOUT}
        t.update(timeouts or {})
        self.timeouts = t

    def url_for(self, endpoint):
        if endpoint not in ALLOWED:
            raise ForbiddenEndpoint(endpoint)
        return f"{self.base}/{endpoint}"

    def _get(self, endpoint, timeout):
        url = self.url_for(endpoint)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                body = r.read(MAX_JPEG + 1)
                return r.status, r.headers.get("Content-Type", ""), body
        except urllib.error.HTTPError as e:
            try:
                body = e.read(4000)
            except Exception:
                body = b""
            raise CameraError(f"/{endpoint}: HTTP {e.code} {body.decode('utf-8', 'replace')[:200]}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise CameraError(f"/{endpoint}: {getattr(e, 'reason', e)}") from None

    def snapshot(self):
        """Свежий кадр JPEG (байты). Не JPEG или обрезан - CameraError."""
        _, _, body = self._get("snap", self.timeouts["snap"])
        if len(body) > MAX_JPEG:
            raise CameraError("/snap: кадр больше 20 МБ")
        if not (body.startswith(b"\xff\xd8") and body.rstrip(b"\r\n\x00").endswith(b"\xff\xd9")):
            raise CameraError("/snap: ответ не JPEG")
        return body

    def status(self) -> PrinterStatus:
        _, _, body = self._get("status", self.timeouts["status"])
        return parse_status(body.decode("utf-8", "replace"))

    def pause(self):
        """Пауза печати. Зовёт только служба с включённой автопаузой (по умолчанию выключена)."""
        _, _, body = self._get("pause", self.timeouts["pause"])
        return body.decode("utf-8", "replace").startswith("ok")
