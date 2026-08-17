#!/usr/bin/env python3
"""Static file server that also accepts rendered frames.

The cinematic page renders deterministically, one frame at a time, and POSTs
each PNG here; that sidesteps needing a screen recorder and gives exact frame
timing regardless of how slow software GL is.

  python3 capture_server.py [port] [outdir]
"""
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8078
OUT = sys.argv[2] if len(sys.argv) > 2 else 'frames'
os.makedirs(OUT, exist_ok=True)


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if not self.path.startswith('/frame'):
            self.send_error(404)
            return
        q = self.path.split('?', 1)[1] if '?' in self.path else ''
        kv = dict(p.split('=', 1) for p in q.split('&') if '=' in p)
        n = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(n)
        name = kv.get('name')
        # the manifests come through this same endpoint; they are not images
        ext = '.json' if self.headers.get('Content-Type', '').startswith(
            'application/json') or data[:1] in (b'{', b'[') else '.png'
        fn = (name + ext) if name else f"f_{int(kv.get('i', 0)):05d}.png"
        with open(os.path.join(OUT, os.path.basename(fn)), 'wb') as f:
            f.write(data)
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'ok')
        print(f'saved {fn} ({len(data)//1024} KB)', flush=True)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    print(f'serving {os.getcwd()} on :{PORT}, frames -> {OUT}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
