#!/usr/bin/env python3
"""
server.py - TokenSafe Remaster Local HTTP Server with Cross-Origin Isolation

Enables WebAssembly multithreading (SharedArrayBuffer) in modern browsers
(Brave, Chrome, Firefox) by serving the required security headers:
  - Cross-Origin-Opener-Policy: same-origin (COOP)
  - Cross-Origin-Embedder-Policy: require-corp (COEP)
  - Access-Control-Allow-Origin: *
"""

import sys
import http.server
import socketserver

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    print(f"============================================================")
    print(f"  TokenSafe Remaster - Servidor Web Local (COOP/COEP Activo)")
    print(f"  URL: http://localhost:{PORT}/                             ")
    print(f"  Modo: Multihilo WASM SIMD a 4 nucleos habilitado           ")
    print(f"============================================================")
    try:
        with ThreadingHTTPServer(("", PORT), CrossOriginIsolatedHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido correctamente.")
        sys.exit(0)
