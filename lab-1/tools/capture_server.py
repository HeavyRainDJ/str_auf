"""Local-only bridge for archiving browser-observed model replies verbatim."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        page = '<!doctype html><meta charset="utf-8"><title>Local lab archive</title><form method="post"><label>Archive JSON<textarea name="payload" rows="12" cols="90"></textarea></label><button type="submit">Save archive</button></form>'
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(page.encode())

    def do_POST(self):
        try:
            if int(self.headers.get('Content-Length', 0)) > 10_000_000:
                raise ValueError('Too large')
            data = parse_qs(self.rfile.read(int(self.headers['Content-Length'])).decode())
            records = json.loads(data['payload'][0])
            for item in records:
                dest = (ROOT / item['path']).resolve()
                if not dest.is_relative_to(ROOT) or dest.exists():
                    raise ValueError('Outside archive or already exists: ' + str(dest))
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(item['content'], encoding='utf-8')
            msg = 'Saved ' + str(len(records)) + ' files'
            self.send_response(200)
        except Exception as exc:
            msg = str(exc)
            self.send_response(400)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(msg.encode())

HTTPServer(('127.0.0.1', 8766), Handler).serve_forever()
