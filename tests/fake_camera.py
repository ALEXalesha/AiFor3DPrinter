"""Поддельный camera_server для тестов: тот же разбор пути, что у настоящего, и журнал запросов.

Настоящий принтер и настоящий camera_server тесты не трогают. Подделка маршрутизирует
путь ровно как camera_server.Handler (self.path.split("?")[0].strip("/")), поэтому
«что услышал бы настоящий сервер» = список fake.routes.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATUS_PRINTING = "🖨 cube.gcode\nСтатус: печатает\nСлой 50/200 (25%)\n🌡 Сопло 215.0°, стол 60°, камера 30°"
STATUS_IDLE = "Статус: простаивает\n🌡 Сопло 25°, стол 24°, камера 24°"


def make_jpeg(w=64, h=36, color=(90, 90, 90)):
    from io import BytesIO

    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


class FakeCameraServer:
    def __init__(self):
        self.raw_paths = []       # как пришло
        self.routes = []          # как разобрал бы camera_server
        self.status_text = STATUS_PRINTING
        self.snap = make_jpeg()
        self.mode = "ok"          # ok / 503 / hang / garbage / drop
        self.hang_s = 2.0
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.raw_paths.append(self.path)
                route = self.path.split("?")[0].strip("/")
                fake.routes.append(route)
                if fake.mode == "hang":
                    time.sleep(fake.hang_s)
                if fake.mode == "drop":
                    self.close_connection = True
                    return
                if fake.mode == "503":
                    return self._send(503, "text/plain; charset=utf-8", "Ошибка: принтер молчит".encode())
                if route == "snap":
                    body = b"not a jpeg" if fake.mode == "garbage" else fake.snap
                    return self._send(200, "image/jpeg", body)
                if route == "status":
                    body = b"\xff\xfe junk" if fake.mode == "garbage" else fake.status_text.encode()
                    return self._send(200, "text/plain; charset=utf-8", body)
                if route in ("pause", "resume", "stopprint"):
                    return self._send(200, "text/plain; charset=utf-8", b"ok ack=0")
                return self._send(404, "text/plain", b"not found")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
