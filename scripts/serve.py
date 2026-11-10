"""Loopback-only DocTrace app. Images are processed in memory, never uploaded externally."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import UnidentifiedImageError, Image
from doctrace.inference import Predictor, decode_image, png_bytes, MAX_BYTES
from doctrace.synthetic import make_document

WEB = Path(__file__).resolve().parents[1] / "web"


def make_handler(predictor, port):
    lock = threading.Lock()
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Do not log filenames or image contents.

        def respond(self, status, body, kind="application/json"):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            if self.headers.get("Host") not in hosts:
                self.respond(403, {"error": "Use the localhost address printed in the terminal."})
                return False
            return True

        def do_GET(self):
            if not self.valid_host():
                return
            route = urlsplit(self.path).path
            if route == "/api/status":
                self.respond(200, predictor.metadata)
            elif route == "/api/sample":
                self.respond(200, png_bytes(make_document(7)), "image/png")
            elif route in ("/", "/app.js", "/style.css"):
                name, kind = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}[route]
                self.respond(200, (WEB / name).read_bytes(), kind)
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if not self.valid_host():
                return
            if self.headers.get("Origin") not in {f"http://{h}" for h in hosts} or self.headers.get("X-DocTrace-Request") != "1":
                self.respond(403, {"error": "This endpoint accepts requests from the local app only."})
                return
            if urlsplit(self.path).path != "/api/analyze":
                self.respond(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    self.respond(413, {"error": "Choose an image smaller than 24 MB."})
                    return
                if not lock.acquire(blocking=False):
                    self.respond(409, {"error": "An analysis is running. Try again when it finishes."})
                    return
                try:
                    self.connection.settimeout(60)
                    image = decode_image(self.rfile.read(length))
                    probe = parse_qs(urlsplit(self.path).query).get("probe", ["1"])[0] == "1"
                    probability, alternative, worst_case, ela_map, report = predictor.analyze(image, probe)
                    encoded = lambda array: base64.b64encode(png_bytes(array)).decode("ascii")
                    self.respond(200, {
                        "image": encoded(image),
                        "probability": encoded((probability * 255).round().astype(np.uint8)),
                        "alternative": encoded((alternative * 255).round().astype(np.uint8)) if alternative is not None else None,
                        "worst_case": encoded((worst_case * 255).round().astype(np.uint8)) if worst_case is not None else None,
                        "ela": encoded(ela_map),
                        "report": report,
                    })
                finally:
                    lock.release()
            except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                self.respond(400, {"error": "Cannot read this image. Use a single PNG, JPEG, or WebP up to 8 megapixels, 8192 pixels per side, and 24 MB."})
            except Exception as error:
                print(f"Analysis failed: {type(error).__name__}: {error}", flush=True)
                self.respond(500, {"error": "Analysis failed. Check the terminal for details; try a smaller crop."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/release_v1/model.pt")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not Path(args.checkpoint).is_file():
        parser.error("Checkpoint missing. Train/export a local model, or pass --checkpoint PATH.")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    predictor = Predictor(args.checkpoint, args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(predictor, args.port))
    print(f"DocTrace is ready at http://127.0.0.1:{args.port} | {predictor.device}. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
