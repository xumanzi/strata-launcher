"""Independent context service. Talks to Strata through HTTP only; never imports or patches it."""
import collections
import copy
import hashlib
import hmac
import http.client
import ipaddress
import json
import os
import select
import socket
import threading
import time
import uuid

import psutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

from strata_memory import ContextCompressor, ContextSummaryError, summary_messages
from strata_usage import UsageTracker
from strata_launch_support import cuda_headroom

CHAT_PATHS = {'/v1/chat/completions', '/v1/messages', '/v1/responses'}
HOP_HEADERS = {'connection', 'transfer-encoding', 'keep-alive', 'proxy-authenticate',
               'proxy-authorization', 'te', 'trailer', 'upgrade'}
MAX_BODY = 64 * 1024 * 1024


class UsageStreamObserver:
    """Observe SSE usage records while returning every proxy byte unchanged."""
    def __init__(self, content_type='text/event-stream', limit=1024 * 1024):
        self.content_type = (content_type or '').lower()
        self.limit = limit
        self.buffer = b''
        self.body = bytearray()
        self.usage = None
        self.complete_usage = False

    def feed(self, data):
        if not data:
            return data
        if 'text/event-stream' in self.content_type:
            self.buffer += data
            while b'\n\n' in self.buffer or b'\r\n\r\n' in self.buffer:
                candidates = [(self.buffer.find(mark), mark) for mark in (b'\r\n\r\n', b'\n\n')
                              if self.buffer.find(mark) >= 0]
                index, mark = min(candidates, key=lambda item: item[0])
                event, self.buffer = self.buffer[:index], self.buffer[index + len(mark):]
                for line in event.splitlines():
                    if not line.startswith(b'data:'):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == b'[DONE]':
                        continue
                    try:
                        value = json.loads(payload)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    found = value.get('usage') if isinstance(value, dict) else None
                    message = value.get('message') if isinstance(value, dict) else None
                    if not isinstance(found, dict) and isinstance(message, dict):
                        found = message.get('usage')
                    if isinstance(found, dict):
                        self.usage = {**(self.usage or {}), **found}
                        if value.get('type') != 'message_start':
                            self.complete_usage |= any(key in found for key in ('completion_tokens', 'output_tokens'))
                    response = value.get('response') if isinstance(value, dict) else None
                    if isinstance(response, dict) and isinstance(response.get('usage'), dict):
                        self.usage = {**(self.usage or {}), **response['usage']}
                        self.complete_usage |= any(key in response['usage'] for key in ('completion_tokens', 'output_tokens'))
        elif len(self.body) < self.limit:
            self.body.extend(data[:self.limit - len(self.body)])
        return data

    def finish(self):
        if self.usage is None and self.body:
            try:
                value = json.loads(self.body)
                if isinstance(value, dict) and isinstance(value.get('usage'), dict):
                    self.usage = value['usage']
                    self.complete_usage = any(key in self.usage for key in ('completion_tokens', 'output_tokens'))
            except (ValueError, UnicodeDecodeError):
                pass
        usage = self.usage or {}
        return (usage, self.complete_usage)


class GatewayError(Exception):
    def __init__(self, message, status=502):
        self.status = status
        super().__init__(message)


def _text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def canonical(path, body):
    """Keep native records for forwarding, with a shared view of complete tool/user turns."""
    if path == '/v1/responses':
        source = body.get('input', [])
        source = [{'role': 'user', 'content': source}] if isinstance(source, str) else source
    else:
        source = body.get('messages', [])
    if not isinstance(source, list) or any(not isinstance(m, dict) for m in source):
        raise GatewayError('messages/input must be a list of objects', 400)
    out = []
    instructions = body.get('system') if path == '/v1/messages' else body.get('instructions') if path == '/v1/responses' else None
    if instructions:
        out.append({'role': 'system', 'content': instructions, '_injected': True})
    for raw in source:
        kind = raw.get('type')
        role = raw.get('role', 'assistant')
        content = raw.get('content', '')
        item = {'role': role, 'content': content, '_raw': raw}
        if kind == 'function_call':
            item.update(role='assistant', tool_calls=[{'id': raw.get('call_id'), 'type': 'function',
                'function': {'name': raw.get('name'), 'arguments': raw.get('arguments')}}])
        elif kind == 'function_call_output':
            item.update(role='tool', content=raw.get('output', ''), tool_call_id=raw.get('call_id'))
        elif kind == 'reasoning':
            item['content'] = raw.get('summary', '')
        if raw.get('reasoning_content'):
            item['reasoning_content'] = raw['reasoning_content']
        if 'tool_calls' in raw:
            item['tool_calls'] = raw['tool_calls']
        if 'tool_call_id' in raw:
            item['tool_call_id'] = raw['tool_call_id']
        if role == 'user' and isinstance(content, list) and content and any(isinstance(p, dict) and p.get('type') == 'tool_result' for p in content):
            item['role'] = 'tool'
        out.append(item)
    return out


def with_messages(path, body, messages):
    updated = dict(body)
    native = []
    for message in messages:
        if message.get('_injected'):
            continue
        if '_raw' in message:
            native.append(message['_raw'])
        elif path == '/v1/responses':
            native.append({'type': 'message', 'role': message['role'], 'content': message['content']})
        else:
            native.append({'role': message['role'], 'content': message['content']})
    updated['input' if path == '/v1/responses' else 'messages'] = native
    return updated


def count_request(messages, body):
    """Use the backend's public tokenizer/template API, with protocol conversion."""
    system, turns = [], []
    for m in messages:
        role, content = m['role'], m.get('content', '')
        if role in ('system', 'developer'):
            system.append(_text(content))
            continue
        blocks = []
        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                kind = p.get('type')
                if kind in ('text', 'input_text', 'output_text'):
                    blocks.append({'type': 'text', 'text': p.get('text', '')})
                elif kind in ('image', 'tool_use', 'tool_result', 'thinking', 'redacted_thinking'):
                    blocks.append(p)
                elif kind in ('image_url', 'input_image'):
                    url = p.get('image_url', '')
                    url = url.get('url', '') if isinstance(url, dict) else url
                    if url.startswith('data:') and ',' in url:
                        meta, data = url.split(',', 1)
                        blocks.append({'type': 'image', 'source': {'type': 'base64', 'media_type': meta[5:].split(';')[0], 'data': data}})
                    else:
                        blocks.append({'type': 'image', 'source': {'type': 'url', 'url': url}})
        elif content:
            blocks.append({'type': 'text', 'text': _text(content)})
        if m.get('reasoning_content'):
            blocks.insert(0, {'type': 'thinking', 'thinking': m['reasoning_content']})
        for c in m.get('tool_calls', []):
            f = c.get('function', {})
            args = f.get('arguments', {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {'raw_arguments': args}
            blocks.append({'type': 'tool_use', 'id': c.get('id') or 'historical-call', 'name': f.get('name'), 'input': args})
        if role == 'tool' and not any(p.get('type') == 'tool_result' for p in blocks):
            blocks = [{'type': 'tool_result', 'tool_use_id': m.get('tool_call_id') or 'historical-call', 'content': content}]
        turns.append({'role': 'assistant' if role == 'assistant' else 'user', 'content': blocks or ''})
    request = {'messages': turns, 'model': body.get('model', ''), 'system': '\n\n'.join(system)}
    tools = []
    for tool in body.get('tools') or []:
        if tool.get('type') not in (None, 'function'):
            continue
        f = tool.get('function', tool)
        tools.append({'name': f.get('name'), 'description': f.get('description', ''),
                      'input_schema': f.get('parameters', f.get('input_schema', {}))})
    if tools:
        request['tools'] = tools
    for key in ('reasoning_effort', 'reasoning_budget_tokens', 'thinking'):
        if key in body:
            request[key] = body[key]
    kw = body.get('chat_template_kwargs') or {}
    if kw.get('enable_thinking') is False or (body.get('reasoning') or {}).get('effort') == 'none':
        request['thinking'] = {'type': 'disabled'}
    return request


def output_limit(path, body):
    key = 'max_output_tokens' if path == '/v1/responses' else 'max_completion_tokens' if 'max_completion_tokens' in body else 'max_tokens'
    value = body.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise GatewayError(key + ' must be an integer', 400)
    return key, int(value)


class ContextGateway:
    def __init__(self, host, port, backend_port, policy, api_key='', cfg_path=None,
                 runtime_config=None, timeout=3600, summary_timeout=300, allowed_hosts=None, trusted_origins=None,
                 log=None, web_root=None, thinking_default='high', llama_backend=False, model_name=''):
        self.backend_port, self.api_key, self.cfg_path = backend_port, api_key, Path(cfg_path) if cfg_path else None
        self.runtime_config = Path(runtime_config) if runtime_config else None
        self.path_replacements = {}
        if self.cfg_path and self.runtime_config:
            source = json.loads(self.cfg_path.read_text(encoding='utf-8-sig'))
            runtime = json.loads(self.runtime_config.read_text(encoding='utf-8-sig'))
            self.path_replacements = {v: u for u, v in zip(source.get('args', []), runtime.get('args', []))
                                      if isinstance(u, str) and isinstance(v, str) and u != v
                                      and PureWindowsPath(u).is_absolute() and PureWindowsPath(v).is_absolute()}
        self.timeout, self.log = timeout, log or (lambda s: None)
        self.summary_timeout = summary_timeout or None
        self.policy_lock, self.last_stats = threading.Lock(), {}
        self.policy = {k: bool(policy.get(k)) for k in ('context_compress_enabled', 'fit_max_tokens')}
        self.compressor = ContextCompressor()
        self.usage = UsageTracker()
        self.context_window = 0
        self.llama_backend, self.model_name = llama_backend, model_name
        self.hardware_lock = threading.Lock()
        self.gpu_sample = (float('-inf'), None)
        self.count_cache, self.count_lock = collections.OrderedDict(), threading.Lock()
        self.allowed_hosts = set(allowed_hosts or []) | {'localhost', '127.0.0.1', '::1', socket.gethostname().lower()}
        self.trusted_origins = set(trusted_origins or [])
        self.web_extension = Path(__file__).with_name('strata_web_extension.js').read_bytes()
        self.code_preview = Path(__file__).with_name('strata_code_preview.js').read_bytes()
        self.monitor_extension = Path(__file__).with_name('strata_monitor_extension.js').read_bytes()
        self.thinking_default = thinking_default
        self.web_root = Path(web_root).resolve() if web_root else None
        if self.web_root and not (self.web_root / 'index.html').is_file():
            raise FileNotFoundError('Strata WebUI is missing: ' + str(self.web_root / 'index.html'))
        gateway = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'  # streamed bodies are delimited by closing the connection
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.handle_request()
            def do_POST(self):
                self.handle_request()
            def do_OPTIONS(self):
                self.handle_request()
            def do_DELETE(self):
                self.handle_request()
            def do_PUT(self):
                self.handle_request()
            def reply(self, value, status=200):
                data = json.dumps(value, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)
            def reply_bytes(self, data, content_type):
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)
            def read_body(self):
                te, length = self.headers.get('Transfer-Encoding'), self.headers.get('Content-Length')
                if te and length:
                    raise GatewayError('ambiguous request framing', 400)
                if te:
                    if te.lower() != 'chunked':
                        raise GatewayError('unsupported Transfer-Encoding', 400)
                    chunks, size = [], 0
                    while True:
                        line = self.rfile.readline(1024)
                        try:
                            n = int(line.split(b';', 1)[0], 16)
                        except ValueError:
                            raise GatewayError('invalid chunk framing', 400) from None
                        if n < 0 or size + n > MAX_BODY:
                            raise GatewayError('request body too large', 413)
                        if not n:
                            for _ in range(100):
                                if self.rfile.readline(8192) == b'\r\n':
                                    return b''.join(chunks)
                            raise GatewayError('invalid chunk trailers', 400)
                        chunk = self.rfile.read(n)
                        if len(chunk) != n or self.rfile.read(2) != b'\r\n':
                            raise GatewayError('incomplete chunk', 400)
                        chunks.append(chunk)
                        size += n
                try:
                    n = int(length or 0)
                except ValueError:
                    raise GatewayError('invalid Content-Length', 400) from None
                if not 0 <= n <= MAX_BODY:
                    raise GatewayError('request body too large', 413)
                data = self.rfile.read(n)
                if len(data) != n:
                    raise GatewayError('incomplete request body', 400)
                return data
            def authorized(self):
                name = urlsplit('//' + self.headers.get('Host', '')).hostname or ''
                try:
                    ipaddress.ip_address(name)
                    host_ok = True
                except ValueError:
                    host_ok = name.lower() in gateway.allowed_hosts
                origin = self.headers.get('Origin', '').rstrip('/')
                own_origin = not origin or urlsplit(origin).netloc == self.headers.get('Host')
                if not host_ok and not gateway.api_key:
                    raise GatewayError('Host is not an allowed server name', 403)
                if origin and not own_origin and origin not in gateway.trusted_origins:
                    raise GatewayError('untrusted browser origin', 403)
                if self.command == 'OPTIONS':
                    return
                public_get = self.command == 'GET' and (urlsplit(self.path).path.rstrip('/') in
                    ('', '/health', '/api/health', '/__strata_launcher/local-key',
                      '/__strata_launcher/web.js', '/__strata_launcher/monitor.js',
                      '/__strata_launcher/code-preview.js', '/__strata_launcher/preferences') or
                     urlsplit(self.path).path.startswith(('/web/', '/fonts/')))
                if gateway.api_key and not public_get:
                    value = self.headers.get('Authorization', '')
                    supplied = value[7:] if value.startswith('Bearer ') else self.headers.get('x-api-key', '')
                    if not hmac.compare_digest(supplied.encode('utf-8'), gateway.api_key.encode('utf-8')):
                        raise GatewayError('invalid or missing API key', 401)
            def handle_request(self):
                cancel, done = threading.Event(), threading.Event()
                started = False
                usage_request_id = None
                usage_status = 500
                usage_result = ({}, False)
                self.connection.settimeout(30)
                try:
                    self.authorized()
                    raw = self.read_body()
                    path = urlsplit(self.path).path.rstrip('/')
                    if path == '/__strata_launcher/local-key':
                        if self.command != 'GET':
                            raise GatewayError('method not allowed', 405)
                        host = (urlsplit('//' + self.headers.get('Host', '')).hostname or '').lower()
                        if host not in ('localhost', '127.0.0.1', '::1') or self.client_address[0] not in ('127.0.0.1', '::1'):
                            raise GatewayError('API key is available only on localhost', 403)
                        return self.reply({'api_key': gateway.api_key})
                    if path == '/__strata_launcher/web.js':
                        if self.command != 'GET':
                            raise GatewayError('method not allowed', 405)
                        return self.reply_bytes(gateway.web_extension, 'application/javascript; charset=utf-8')
                    if path == '/__strata_launcher/monitor.js':
                        if self.command != 'GET':
                            raise GatewayError('method not allowed', 405)
                        return self.reply_bytes(gateway.monitor_extension, 'application/javascript; charset=utf-8')
                    if path == '/__strata_launcher/code-preview.js':
                        if self.command != 'GET':
                            raise GatewayError('method not allowed', 405)
                        return self.reply_bytes(gateway.code_preview, 'application/javascript; charset=utf-8')
                    if path == '/__strata_launcher/preferences':
                        if self.command != 'GET':
                            raise GatewayError('method not allowed', 405)
                        return self.reply({'reasoning_effort': gateway.thinking_default})
                    if path == '/__strata_launcher/usage':
                        if self.command != 'GET':
                            raise GatewayError('method not allowed', 405)
                        return self.reply(gateway.usage_snapshot(cancel))
                    if path == '/__strata_launcher/usage/reset':
                        if self.command != 'POST':
                            raise GatewayError('method not allowed', 405)
                        gateway.usage.reset()
                        return self.reply(gateway.usage.snapshot(gateway.context_window))
                    if gateway.web_root and self.command == 'GET' and (path == '' or
                            path.startswith(('/web/', '/fonts/'))):
                        if path == '':
                            page = (gateway.web_root / 'index.html').read_bytes()
                            scripts = (b'<script src="/__strata_launcher/code-preview.js"></script>\n'
                                       b'<script src="/__strata_launcher/web.js"></script>\n'
                                       b'<script src="/__strata_launcher/monitor.js"></script>\n</body>')
                            page = page.replace(b'</body>', scripts, 1)
                            return self.reply_bytes(page, 'text/html; charset=utf-8')
                        relative = path.removeprefix('/web/') if path.startswith('/web/') else 'fonts/' + path.removeprefix('/fonts/')
                        file = (gateway.web_root / relative).resolve()
                        if not file.is_relative_to(gateway.web_root) or not file.is_file():
                            raise GatewayError('WebUI file not found', 404)
                        mime = {'.js': 'application/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
                                '.svg': 'image/svg+xml', '.woff2': 'font/woff2'}.get(file.suffix.lower(),
                                'application/octet-stream')
                        return self.reply_bytes(file.read_bytes(), mime)
                    if path == '/__strata_context/config':
                        if not gateway.api_key and self.client_address[0] not in ('127.0.0.1', '::1'):
                            raise GatewayError('context control requires an API key or a local client', 403)
                        if self.command == 'POST':
                            gateway.update_policy(json.loads(raw).get('set', {}))
                        elif self.command != 'GET':
                            raise GatewayError('method not allowed', 405)
                        return self.reply(gateway.report())
                    if gateway.launch_ready is not None and not gateway.launch_ready.is_set():
                        raise GatewayError('Model startup and GPU memory verification in progress', 503)
                    if gateway.llama_backend and self.command == 'GET' and path == '/metrics':
                        return self.reply(gateway.llama_metrics())
                    def watch():
                        while not done.wait(.2):
                            try:
                                readable, _, _ = select.select([self.connection], [], [], 0)
                                if readable and self.connection.recv(1, socket.MSG_PEEK) == b'':
                                    cancel.set()
                                    return
                            except OSError:
                                cancel.set()
                                return
                    threading.Thread(target=watch, daemon=True).start()
                    with gateway.policy_lock:
                        policy = dict(gateway.policy)
                    headers = gateway.forward_headers(self.headers)
                    body = None
                    if self.command == 'POST' and path in CHAT_PATHS:
                        body = json.loads(raw)
                        usage_request_id = uuid.uuid4().hex
                        gateway.usage.begin(usage_request_id, path, str(body.get('model', '')))
                        if any(policy.values()):
                            body = gateway.prepare(path, body, policy, cancel)
                            raw = json.dumps(body, ensure_ascii=False).encode()
                        gateway.usage.mark_forwarded(usage_request_id)
                    conn, response = gateway.connect(self.command, self.path, raw, headers, cancel)
                    try:
                        usage_status = response.status
                        # A native context rejection gives the exact count for its own protocol.
                        if response.status == 400 and self.command == 'POST' and path in CHAT_PATHS and policy['fit_max_tokens']:
                            error = response.read()
                            import re
                            match = re.search(rb'prompt \((\d+) tokens\).*?context \((\d+)\)', error)
                            if match:
                                prompt, context = map(int, match.groups())
                                room = context - prompt - 8
                                key, requested = output_limit(path, body)
                                if room > 0 and requested > room:
                                    conn.close()
                                    body[key] = room
                                    conn, response = gateway.connect(self.command, self.path,
                                        json.dumps(body, ensure_ascii=False).encode(), headers, cancel)
                                else:
                                    return self.raw_reply(response, error)
                            else:
                                return self.raw_reply(response, error)
                        if self.command == 'POST' and response.status == 200:
                            if path == '/config':
                                gateway.mirror_config(json.loads(raw))
                            elif path == '/settings':
                                gateway.mirror_settings()
                        if path == '/config' and response.status == 200 and gateway.cfg_path:
                            view = json.loads(response.read())
                            view['file'] = gateway.cfg_path.name
                            saved = json.loads(gateway.cfg_path.read_text(encoding='utf-8-sig'))
                            for entry in view.get('keys', []):
                                if entry.get('key') == 'open_browser':
                                    entry['value'] = saved.get('open_browser')
                                if entry.get('key') == 'fit_max_tokens':
                                    entry['value'] = gateway.report()['live']['fit_max_tokens']
                            return self.reply(view)
                        self.send_response(response.status)
                        connection_headers = {s.strip().lower() for s in response.getheader('Connection', '').split(',')}
                        for k, v in response.getheaders():
                            if k.lower() not in HOP_HEADERS | connection_headers:
                                self.send_header(k, v)
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        started = True
                        observer = UsageStreamObserver(response.getheader('Content-Type', '')) if usage_request_id else None
                        while not cancel.is_set():
                            data = response.read1(16384)
                            if not data:
                                break
                            if observer:
                                observer.feed(data)
                                if observer.usage:
                                    gateway.usage.progress(usage_request_id, observer.usage)
                            self.wfile.write(data)
                            self.wfile.flush()
                        if observer:
                            usage_result = observer.finish()
                    finally:
                        gateway.close_connection(conn)
                except (GatewayError, ContextSummaryError) as e:
                    if not started and not cancel.is_set():
                        self.reply({'error': {'type': 'context_service_error', 'message': str(e)}}, getattr(e, 'status', 502))
                except (ValueError, TypeError, KeyError) as e:
                    if not started and not cancel.is_set():
                        self.reply({'error': {'type': 'invalid_request_error', 'message': str(e)}}, 400)
                except (OSError, http.client.HTTPException) as e:
                    if not started and not cancel.is_set():
                        try:
                            self.reply({'error': {'type': 'backend_unavailable', 'message': 'Strata backend is unavailable or timed out'}}, 503)
                        except OSError:
                            pass
                finally:
                    if usage_request_id:
                        gateway.usage.finish(usage_request_id, usage_result[0], usage_status, usage_result[1])
                    done.set()
                    cancel.set()
                    with gateway.events_lock:
                        gateway.cancel_events.discard(cancel)
                    self.close_connection = True
            def raw_reply(self, response, error):
                self.send_response(response.status)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                self.send_header('Content-Length', str(len(error)))
                self.end_headers()
                self.wfile.write(error)
        class Server(ThreadingHTTPServer):
            allow_reuse_address = os.name != 'nt'
            daemon_threads = True
        if ':' in host:
            Server.address_family = socket.AF_INET6
        self.server = Server((host, port), Handler)
        self.port = self.server.server_port
        self.cancel_events, self.events_lock = set(), threading.Lock()
        self.started = False
        self.launch_ready = None

    def start(self):
        self.started = True
        threading.Thread(target=self.server.serve_forever, daemon=True, name='strata-context-service').start()

    def stop(self):
        with self.events_lock:
            for event in self.cancel_events:
                event.set()
        if self.started:
            self.server.shutdown()
        self.server.server_close()

    def update_policy(self, policy):
        if set(policy) != {'context_compress_enabled', 'fit_max_tokens'} or any(not isinstance(v, bool) for v in policy.values()):
            raise GatewayError('both context flags must be booleans', 400)
        with self.policy_lock:
            if self.cfg_path:
                self.save_policy(self.cfg_path, policy)
            self.policy = dict(policy)
        return self.report()

    @staticmethod
    def save_policy(path, policy):
        path = Path(path)
        cfg = json.loads(path.read_text(encoding='utf-8-sig'))
        if all(cfg.get(k) == v for k, v in policy.items()):
            return
        import shutil
        shutil.copy2(path, str(path) + '.context.bak')
        cfg.update(policy)
        tmp = path.with_name(path.name + '.context.tmp')
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, path)

    def report(self):
        with self.policy_lock:
            return {'live': dict(self.policy), 'context_guard': dict(self.last_stats), 'service': 'independent-context'}

    def forward_headers(self, headers):
        connection_headers = {s.strip().lower() for s in headers.get('Connection', '').split(',')}
        out = {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS | connection_headers |
               {'host', 'content-length', 'x-strata-internal-summary'}}
        origin = headers.get('Origin', '').rstrip('/')
        if origin and urlsplit(origin).netloc == headers.get('Host'):
            out['Origin'] = 'http://127.0.0.1:%d' % self.backend_port
        if self.api_key:
            out['Authorization'] = 'Bearer ' + self.api_key
        return out

    @staticmethod
    def close_connection(conn):
        if hasattr(conn, '_context_done'):
            conn._context_done.set()
        sock = getattr(conn, '_context_socket', conn.sock)
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        conn.close()
        response = getattr(conn, '_context_response', None)
        if response:
            response.close()

    def connect(self, method, path, raw, headers, cancel, timeout=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.backend_port, timeout=timeout if timeout is not None else self.timeout)
        ended = threading.Event()
        conn._context_done = ended
        def watch():
            while not ended.wait(.1):
                if cancel.is_set():
                    self.close_connection(conn)
                    return
        with self.events_lock:
            self.cancel_events.add(cancel)
        try:
            if cancel.is_set():
                raise GatewayError('request cancelled', 499)
            conn.connect()
            conn._context_socket = conn.sock
            threading.Thread(target=watch, daemon=True).start()
            conn.request(method, path, body=raw if method not in ('GET', 'OPTIONS') else None, headers=headers)
            response = conn.getresponse()
            conn._context_response = response
            return conn, response
        except Exception:
            self.close_connection(conn)
            raise

    def json_backend(self, path, body=None, cancel=None, summary=False, timeout=None):
        cancel = cancel if cancel is not None else threading.Event()
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        if summary:
            headers['X-Strata-Internal-Summary'] = '1'
        raw = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        timeout = timeout if timeout is not None else self.summary_timeout if summary else 30
        conn, response = self.connect('POST' if body is not None else 'GET', path, raw, headers, cancel, timeout=timeout or self.timeout)
        try:
            data = response.read()
            if response.status != 200:
                raise GatewayError('Strata public API unavailable: ' + path + ' (HTTP %d)' % response.status, 503)
            return json.loads(data)
        finally:
            self.close_connection(conn)

    def count(self, messages, body, cancel):
        request = count_request(messages, body)
        digest = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self.count_lock:
            if digest in self.count_cache:
                self.count_cache.move_to_end(digest)
                return self.count_cache[digest]
        value = self.json_backend('/v1/messages/count_tokens', request, cancel).get('input_tokens')
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GatewayError('invalid token count returned by Strata')
        # The public count API uses Anthropic rendering. Reserve space for protocol-specific wrappers.
        value += 128
        with self.count_lock:
            self.count_cache[digest] = value
            while len(self.count_cache) > 128:
                self.count_cache.popitem(last=False)
        return value

    def usage_snapshot(self, cancel=None):
        if not self.context_window:
            try:
                props = self.json_backend('/props', cancel=cancel, timeout=2)
                context = props.get('default_generation_settings', {}).get('n_ctx', 0)
                if isinstance(context, int) and context > 0:
                    self.context_window = context
            except (GatewayError, OSError, http.client.HTTPException, ValueError, TypeError):
                pass
        requests = self.usage.forwarded_requests()
        # /metrics describes engine work, not a gateway request ID. Only attribute it
        # when one user request is in flight; parallel streams retain their own usage.
        if len(requests) == 1 and not self.llama_backend:
            try:
                metrics = self.json_backend('/metrics', cancel=cancel, timeout=2)
                live = metrics.get('live') or {}
                if (live.get('state') in ('reading', 'generating', 'decoding')
                        and int(live.get('running') or 1) == 1
                        and isinstance(live.get('prompt_tokens'), (int, float))):
                    self.usage.progress(requests[0], {'input_tokens': live['prompt_tokens'],
                        'output_tokens': live.get('generated') or 0}, source='engine')
            except (GatewayError, OSError, http.client.HTTPException, ValueError, TypeError, AttributeError):
                pass
        return self.usage.snapshot(self.context_window)

    def llama_metrics(self):
        snapshot = self.usage_snapshot()
        memory = psutil.virtual_memory()
        hardware = {'ram_total': memory.total, 'ram_used': memory.total - memory.available}
        with self.hardware_lock:
            now = time.monotonic()
            sampled, memory = self.gpu_sample
            if now - sampled >= 5:
                memory = cuda_headroom()
                self.gpu_sample = (now, memory)
        if memory is not None:
            free, total = memory
            hardware.update(gpu_mem_used=(total - free) * 1024**2, gpu_mem_total=total * 1024**2)

        last = snapshot['last_request']
        recent = []
        if last is not None:
            recent.append({
                'id': last['finished_at'], 'time': last['finished_at'],
                'prompt_tokens': last['input_tokens'] if last['precise'] else None,
                'output_tokens': last['output_tokens'] if last['precise'] else None,
                'finish': 'error' if last['status'] >= 400 else None,
            })
        running = len(self.usage.forwarded_requests())
        current = snapshot.get('current_request') or {}
        return {
            'live': {'state': 'generating' if running else 'idle', 'running': running,
                     'generated': current.get('output_tokens'), 'tok_s': None},
            'engine': {'model': self.model_name, 'max_context': self.context_window},
            'hardware': hardware,
            'requests': recent, 'requests_kept': len(recent),
        }

    def summarize(self, payload, cap, model, cancel):
        result = self.json_backend('/v1/chat/completions', {'model': model, 'messages': summary_messages(payload),
            'max_tokens': cap, 'temperature': .2, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}, cancel, summary=True)
        choices = result.get('choices') or []
        if not choices or choices[0].get('finish_reason') != 'stop':
            raise GatewayError('summary generation did not finish completely; original history preserved')
        text = choices[0].get('message', {}).get('content')
        if not isinstance(text, str) or not text.strip():
            raise GatewayError('summary generation returned empty memory; original history preserved')
        return text.strip()

    def prepare(self, path, body, policy, cancel):
        if not isinstance(body, dict):
            raise GatewayError('request must be a JSON object', 400)
        if path == '/v1/responses' and (body.get('previous_response_id') or body.get('conversation')):
            raise GatewayError('context service requires stateless Responses input with full history', 400)
        try:
            props = self.json_backend('/props', cancel=cancel)
        except GatewayError as e:
            if e.status != 503:
                raise
            self.json_backend('/load', {}, cancel)
            props = self.json_backend('/props', cancel=cancel)
        context = props.get('default_generation_settings', {}).get('n_ctx', 0)
        if context == 0:
            self.json_backend('/load', {}, cancel)
            props = self.json_backend('/props', cancel=cancel)
            context = props.get('default_generation_settings', {}).get('n_ctx', 0)
        if not isinstance(context, int) or context <= 0:
            raise GatewayError('Strata engine is starting; context window is not ready', 503)
        self.context_window = context
        messages = canonical(path, body)
        key, maximum = output_limit(path, body)
        count = lambda ms: self.count(ms, body, cancel)
        if policy['context_compress_enabled']:
            model = body.get('model', '')
            summary_body = {'model': model, 'thinking': {'type': 'disabled'}}
            messages, stats = self.compressor.compact(messages, count, context, maximum,
                lambda payload, cap: self.summarize(payload, cap, model, cancel),
                lambda payload: self.count(summary_messages(payload), summary_body, cancel), cancel,
                cache_scope=[model, props.get('model_alias'), body.get('tools'), props.get('chat_template')])
        else:
            n = count(messages)
            stats = {'enabled': False, 'prompt_before': n, 'prompt_after': n, 'summarized_messages': 0}
        updated = with_messages(path, body, messages) if stats.get('summarized_messages') else dict(body)
        room = context - count(messages) - 8
        if policy['fit_max_tokens'] and maximum > 0 and maximum > room:
            if room < 1:
                raise GatewayError('current input leaves no room in the context; latest turn was not truncated', 400)
            updated[key] = room
        stats['effective_output'] = updated.get(key, maximum)
        with self.policy_lock:
            self.last_stats = dict(stats)
        self.usage.compression(stats, stats.get('summary_cached'))
        if stats.get('summarized_messages'):
            self.log('[context] memory: %s -> %s tokens, %s messages, cached=%s' %
                (stats['prompt_before'], stats['prompt_after'], stats['summarized_messages'], stats.get('summary_cached')))
        return updated

    def mirror_config(self, request):
        if not self.cfg_path or not self.runtime_config:
            return
        with self.policy_lock:
            saved = json.loads(self.cfg_path.read_text(encoding='utf-8-sig'))
            runtime = json.loads(self.runtime_config.read_text(encoding='utf-8-sig'))
            protected = {'host', 'port', 'context_compress_enabled', 'fit_max_tokens', 'open_browser',
                         'exe', 'cwd', 'tokenizer', 'log', 'vision'}
            for key in list(saved):
                if key not in protected and key not in runtime:
                    saved.pop(key)
            saved.update({k: v for k, v in runtime.items() if k not in protected})
            if isinstance(saved.get('args'), list):
                saved['args'] = [self.path_replacements.get(arg, arg) if isinstance(arg, str) else arg
                                 for arg in saved['args']]
            for key in ('host', 'port', 'open_browser'):
                if key in request.get('set', {}):
                    if request['set'][key] is None:
                        saved.pop(key, None)
                    else:
                        saved[key] = request['set'][key]
            if 'fit_max_tokens' in request.get('set', {}):
                self.policy['fit_max_tokens'] = runtime.get('fit_max_tokens') is True
                saved['fit_max_tokens'] = self.policy['fit_max_tokens']
                runtime['fit_max_tokens'] = False
                self.runtime_config.write_text(json.dumps(runtime, ensure_ascii=False, indent=1), encoding='utf-8')
            import shutil
            shutil.copy2(self.cfg_path, str(self.cfg_path) + '.gateway.bak')
            tmp = self.cfg_path.with_name(self.cfg_path.name + '.gateway.tmp')
            tmp.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding='utf-8')
            os.replace(tmp, self.cfg_path)

    def mirror_settings(self):
        if not self.cfg_path or not self.runtime_config:
            return
        import shutil
        source, target = self.runtime_config.with_suffix('.shared-settings.json'), self.cfg_path.with_suffix('.shared-settings.json')
        with self.policy_lock:
            if source.is_file():
                tmp = target.with_name(target.name + '.gateway.tmp')
                shutil.copy2(source, tmp)
                os.replace(tmp, target)
            else:
                target.unlink(missing_ok=True)
