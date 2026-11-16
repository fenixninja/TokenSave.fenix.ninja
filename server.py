#!/usr/bin/env python3
"""
server.py - TokenSafe Remaster Local HTTP Server with Cross-Origin Isolation

Enables WebAssembly multithreading (SharedArrayBuffer) in modern browsers
(Brave, Chrome, Firefox) by serving the required security headers:
  - Cross-Origin-Opener-Policy: same-origin (COOP)
  - Cross-Origin-Embedder-Policy: credentialless (COEP)
  - Cross-Origin-Resource-Policy: cross-origin (CORP)
  - Access-Control-Allow-Origin: *
"""

import sys
import os
import http.server
import socketserver
from datetime import datetime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Gracefully handle browser favicon and DevTools probes without 404 clutter
        if self.path in ("/favicon.ico", "/favicon.png") or self.path.startswith("/.well-known/"):
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        now = datetime.now().strftime("%H:%M:%S")
        try:
            msg = format % args
        except Exception:
            msg = " ".join(str(a) for a in args)

        # Format output with clear color markers safely on the formatted string
        if "/models/" in msg:
            color = "\033[1;36m"  # Cyan
            tag = "[MODEL]"
        elif " 200 " in f" {msg} ":
            color = "\033[32m"    # Green
            tag = "[OK 200]"
        elif " 304 " in f" {msg} ":
            color = "\033[33m"    # Yellow
            tag = "[CACHE]"
        elif any(e in msg for e in [" 404 ", "code 404"]):
            color = "\033[1;31m"  # Red
            tag = "[404]"
        else:
            color = "\033[0m"
            tag = "[HTTP]"
        reset = "\033[0m"

        print(f"[{now}] {color}{tag}{reset} {msg}", flush=True)

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    print(f"============================================================")
    print(f"  TokenSafe Remaster - Servidor Web Local (COOP/COEP Activo)")
    print(f"  URL: http://localhost:{PORT}/                             ")
    print(f"  Directorio raiz: {os.getcwd()}                             ")
    print(f"  Modo: Multihilo WASM SIMD (SharedArrayBuffer) habilitado   ")
    print(f"============================================================", flush=True)
    try:
        with ThreadingHTTPServer(("", PORT), CrossOriginIsolatedHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido correctamente.", flush=True)
        sys.exit(0)
