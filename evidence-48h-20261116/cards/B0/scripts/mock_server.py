#!/usr/bin/env python3
"""CPU mock of llama-server for dry-run validation: /health /tokenize /completion(stream) /slots /metrics. Deterministic from the prompt tail. usage: mock_server.py PORT"""
import sys, os, json, hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def _j(s, o, c=200): d = json.dumps(o).encode(); s.send_response(c); s.send_header('Content-Type', 'application/json'); s.send_header('Content-Length', str(len(d))); s.end_headers(); s.wfile.write(d)
    def do_GET(s):
        if s.path == '/health': return s._j({'status': 'ok'})
        if s.path == '/metrics': d = b'llamacpp:draft_tokens_total 5\n'; s.send_response(200); s.send_header('Content-Length', str(len(d))); s.end_headers(); return s.wfile.write(d)
        s._j({}, 404)
    def do_POST(s):
        b = json.loads(s.rfile.read(int(s.headers['Content-Length'])) or b'{}')
        if s.path == '/tokenize': t = b['content'].encode(); return s._j({'tokens': [int.from_bytes(hashlib.md5(t[i:i + 3]).digest()[:2], 'big') + 1000 for i in range(0, len(t), 3)]})
        if s.path.startswith('/slots'): return s._j({'n_restored': 0}, 404 if 'restore' in s.path else 200)
        p = b['prompt']; h = int(hashlib.sha256(json.dumps(p[-64:]).encode()).hexdigest()[:8], 16); n = min(b['n_predict'], (3 if os.environ.get('MOCK_SHORT') else 80) + h % 700); eos = n < b['n_predict']
        s.send_response(200); s.send_header('Content-Type', 'text/event-stream'); s.end_headers()
        for i in range(n): s.wfile.write(b'data: ' + json.dumps({'content': 'x', 'tokens': [(h + i) % 50000]}).encode() + b'\n\n'); s.wfile.flush()
        s.wfile.write(b'data: ' + json.dumps({'stop': True, 'stop_type': 'eos' if eos else 'limit', 'tokens_predicted': n, 'tokens_evaluated': len(p), 'truncated': False,
            'timings': {'prompt_n': 8, 'cache_n': len(p) - 8, 'prompt_ms': 5, 'predicted_ms': n, 'predicted_per_second': 1000, 'draft_n': n // 2, 'draft_n_accepted': n // 4}}).encode() + b'\n\n')
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
