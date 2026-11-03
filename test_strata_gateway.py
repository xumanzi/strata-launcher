import http.client
import json
import threading
import time
import tempfile
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from strata_gateway import ContextGateway, UsageStreamObserver


class Backend:
    def __init__(self):
        self.requests = []
        self.summary_calls = 0
        self.summary_finish = 'stop'
        self.summary_text = 'goal=fix startup; checked engine; pending=benchmark'
        self.context = 4096
        self.cancelled = threading.Event()
        backend = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def do_GET(self):
                if self.path == '/':
                    data = b'<html><body><main>Strata chat</main></body></html>'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if self.path == '/props':
                    return self.reply({'default_generation_settings': {'n_ctx': backend.context}})
                self.reply({'status': 'ok'})
            def reply(self, value, status=200):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                body = json.loads(raw)
                if self.path == '/v1/messages/count_tokens':
                    total = len(json.dumps(body.get('messages', []), ensure_ascii=False)) + len(str(body.get('system', '')))
                    return self.reply({'input_tokens': total})
                if self.headers.get('X-Strata-Internal-Summary') == '1':
                    backend.summary_calls += 1
                    return self.reply({'choices': [{'message': {'content': backend.summary_text},
                        'finish_reason': backend.summary_finish}]})
                backend.requests.append((self.path, raw, body, dict(self.headers)))
                if body.get('stream'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.end_headers()
                    try:
                        for data in [b'data: {"delta":"hello"}\n\n',
                                     b'data: {"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}\n\n',
                                     b'data: [DONE]\n\n']:
                            self.wfile.write(('%x\r\n' % len(data)).encode() + data + b'\r\n')
                            self.wfile.flush()
                        self.wfile.write(b'0\r\n\r\n')
                    except OSError:
                        backend.cancelled.set()
                else:
                    self.reply({'ok': True, 'usage': {'prompt_tokens': 9, 'completion_tokens': 2,
                                                      'total_tokens': 11}})
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.gateway = ContextGateway('127.0.0.1', 0, self.backend.server.server_port,
            {'context_compress_enabled': True, 'fit_max_tokens': True}, api_key='test-key')
        self.gateway.start()
        self.messages = [{'role': 'user', 'content': 'goal=fix startup; ' + 'x' * 1700},
                         {'role': 'assistant', 'content': 'checked engine; ' + 'y' * 1000},
                         {'role': 'user', 'content': 'continue'}]

    def tearDown(self):
        self.gateway.stop()
        self.backend.stop()

    def request(self, path, body=None, auth=True, method='POST', headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.gateway.port, timeout=10)
        h = {'Content-Type': 'application/json', **(headers or {})}
        if auth:
            h['Authorization'] = 'Bearer test-key'
        raw = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=raw, headers=h)
        response = conn.getresponse()
        status, result_headers, data = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return status, result_headers, data

    def test_all_three_native_protocols_receive_summary_and_recent_turn(self):
        for path, body in [('/v1/chat/completions', {'messages': self.messages, 'max_tokens': 1800}),
                           ('/v1/messages', {'messages': self.messages, 'max_tokens': 1800}),
                           ('/v1/responses', {'input': self.messages, 'max_output_tokens': 1800})]:
            with self.subTest(path=path):
                self.assertEqual(self.request(path, body)[0], 200)
                final = self.backend.requests[-1][2]
                text = json.dumps(final, ensure_ascii=False)
                self.assertIn('pending=benchmark', text)
                self.assertIn('continue', text)
                self.assertNotIn('x' * 100, text)
        self.assertGreater(self.backend.summary_calls, 0)
        self.assertEqual(len(self.messages), 3)

    def test_disabled_policy_forwards_exact_original_body(self):
        self.gateway.update_policy({'context_compress_enabled': False, 'fit_max_tokens': False})
        raw = b'{ "messages": [{"role":"user","content":"hello"}], "max_tokens": 12 }'
        self.assertEqual(self.request('/v1/chat/completions', raw)[0], 200)
        self.assertEqual(self.backend.requests[-1][1], raw)
        self.assertEqual(self.backend.summary_calls, 0)

    def test_chunked_sse_is_forwarded_as_stream_data_with_valid_framing(self):
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 128, 'stream': True}
        status, headers, data = self.request('/v1/chat/completions', body)
        self.assertEqual(status, 200)
        self.assertNotIn('Transfer-Encoding', headers)
        self.assertEqual(data, b'data: {"delta":"hello"}\n\n'
                               b'data: {"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}\n\n'
                               b'data: [DONE]\n\n')

    def test_stream_observer_reassembles_split_sse_without_changing_bytes(self):
        raw = (b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
               b'data: {"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}\n\n'
               b'data: [DONE]\n\n')
        observer = UsageStreamObserver()
        chunks = [raw[:17], raw[17:63], raw[63:91], raw[91:]]
        forwarded = b''.join(observer.feed(chunk) for chunk in chunks)
        self.assertEqual(forwarded, raw)
        self.assertEqual(observer.finish(), ({'prompt_tokens': 12, 'completion_tokens': 3,
                                               'total_tokens': 15}, True))

    def test_stream_observer_accepts_crlf_and_merges_anthropic_usage_events(self):
        raw = (b'data: {"type":"message_start","message":{"usage":{"input_tokens":18}}}\r\n\r\n'
               b'data: {"type":"message_delta","usage":{"output_tokens":7}}\r\n\r\n')
        observer = UsageStreamObserver()
        self.assertEqual(observer.feed(raw[:43]) + observer.feed(raw[43:]), raw)
        usage, precise = observer.finish()
        self.assertTrue(precise)
        self.assertEqual(usage, {'input_tokens': 18, 'output_tokens': 7})

    def test_usage_endpoint_reports_all_chat_clients_and_can_reset(self):
        self.gateway.update_policy({'context_compress_enabled': False, 'fit_max_tokens': False})
        self.assertEqual(self.request('/v1/chat/completions',
            {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8})[0], 200)
        self.assertEqual(self.request('/v1/messages',
            {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8, 'stream': True})[0], 200)
        status, _, raw = self.request('/__strata_launcher/usage', auth=False, method='GET')
        self.assertEqual(status, 401)
        status, _, raw = self.request('/__strata_launcher/usage', method='GET')
        usage = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertEqual(usage['requests']['successful'], 2)
        self.assertEqual(usage['totals'], {'input_tokens': 21, 'output_tokens': 5, 'total_tokens': 26})
        self.assertEqual(usage['context']['limit_tokens'], self.backend.context)
        status, _, raw = self.request('/__strata_launcher/usage/reset', body={})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['requests']['total'], 0)

    def test_authentication_is_required_before_forwarding(self):
        self.assertEqual(self.request('/v1/chat/completions', {'messages': []}, auth=False)[0], 401)
        self.assertFalse(self.backend.requests)

    def test_empty_summary_fails_without_forwarding_shortened_history(self):
        self.backend.summary_text = ''
        status, _, raw = self.request('/v1/chat/completions', {'messages': self.messages, 'max_tokens': 1800})
        self.assertEqual(status, 502)
        self.assertIn('summary', json.loads(raw)['error']['message'].lower())
        self.assertFalse(self.backend.requests)

    def test_incomplete_summary_is_not_used(self):
        self.backend.summary_finish = 'length'
        self.assertEqual(self.request('/v1/messages', {'messages': self.messages, 'max_tokens': 1800})[0], 502)
        self.assertFalse(self.backend.requests)

    def test_repeated_history_reuses_summary(self):
        body = {'messages': self.messages, 'max_tokens': 1800}
        self.request('/v1/chat/completions', body)
        n = self.backend.summary_calls
        self.request('/v1/chat/completions', body)
        self.assertEqual(self.backend.summary_calls, n)
        self.assertTrue(self.gateway.report()['context_guard']['summary_cached'])

    def test_fit_output_preserves_large_single_input(self):
        self.backend.context = 131072
        body = {'messages': [{'role': 'user', 'content': 'x' * 99718}], 'max_tokens': 32768}
        self.assertEqual(self.request('/v1/chat/completions', body)[0], 200)
        final = self.backend.requests[-1][2]
        self.assertEqual(final['messages'], body['messages'])
        self.assertLessEqual(final['max_tokens'], 31346)
        self.assertEqual(self.backend.summary_calls, 0)

    def test_anthropic_recent_tool_use_and_result_remain_paired(self):
        messages = self.messages[:-1] + [{'role': 'user', 'content': 'latest'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'c1', 'name': 'read_file', 'input': {'path': 'a.py'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'c1', 'content': 'important result'}]}]
        self.assertEqual(self.request('/v1/messages', {'messages': messages, 'max_tokens': 1800})[0], 200)
        final = self.backend.requests[-1][2]['messages']
        self.assertEqual(final[-3:], messages[-3:])

    def test_control_endpoint_updates_policy_and_is_authenticated(self):
        body = {'set': {'context_compress_enabled': False, 'fit_max_tokens': True}}
        path = '/__strata_context/config'
        self.assertEqual(self.request(path, body, auth=False)[0], 401)
        status, _, raw = self.request(path, body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['live'], body['set'])

    def test_browser_bootstrap_public_but_foreign_origin_rejected(self):
        self.assertEqual(self.request('/', auth=False, method='GET')[0], 200)
        self.assertEqual(self.request('/health', auth=False, method='GET')[0], 200)
        self.assertEqual(self.request('/v1/chat/completions', {'messages': []},
            headers={'Origin': 'http://foreign.example'})[0], 403)
        self.assertEqual(self.backend.requests, [])

    def test_local_webui_receives_key_but_lan_host_does_not(self):
        path = '/__strata_launcher/local-key'
        status, _, raw = self.request(path, auth=False, method='GET')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), {'api_key': 'test-key'})
        self.assertEqual(self.request(path, auth=False, method='GET',
            headers={'Host': '192.168.1.85:%d' % self.gateway.port})[0], 403)
        self.assertEqual(self.request(path, auth=False, method='GET',
            headers={'Origin': 'http://foreign.example'})[0], 403)
        self.assertEqual(self.request(path, auth=True, method='POST')[0], 405)

    def test_unconfigured_gateway_does_not_modify_backend_html(self):
        status, _, page = self.request('/', auth=False, method='GET')
        self.assertEqual(status, 200)
        self.assertIn(b'<main>Strata chat</main>', page)
        self.assertNotIn(b'/__strata_launcher/web.js', page)
        status, headers, script = self.request('/__strata_launcher/web.js', auth=False, method='GET')
        self.assertEqual(status, 200)
        self.assertIn('javascript', headers.get('Content-Type', ''))
        self.assertIn(b'local-key', script)

    def test_installed_strata_webui_is_served_at_localhost_and_lan_host(self):
        with tempfile.TemporaryDirectory() as directory:
            web = Path(directory)
            (web / 'index.html').write_text('<html><body><h1>Strata installed UI</h1></body></html>')
            (web / 'monitor.html').write_text('<html><body><h1>API Monitor</h1></body></html>')
            (web / 'app.js').write_text('window.strataInstalled = true;')
            gateway = ContextGateway('127.0.0.1', 0, self.backend.server.server_port,
                {'context_compress_enabled': False, 'fit_max_tokens': False},
                api_key='test-key', web_root=web)
            gateway.start()
            try:
                for host in ('127.0.0.1', '192.168.1.85'):
                    conn = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=5)
                    conn.request('GET', '/', headers={'Host': f'{host}:{gateway.port}'})
                    response = conn.getresponse()
                    page, headers = response.read(), dict(response.getheaders())
                    conn.close()
                    self.assertEqual(response.status, 200)
                    self.assertIn(b'Strata installed UI', page)
                    self.assertIn(b'/__strata_launcher/web.js', page)
                    self.assertIn(b'/__strata_launcher/code-preview.js', page)
                    self.assertEqual(headers.get('Cache-Control'), 'no-store')
                conn = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=5)
                conn.request('GET', '/web/app.js')
                response = conn.getresponse()
                self.assertEqual(response.read(), b'window.strataInstalled = true;')
                conn.close()
                conn = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=5)
                conn.request('GET', '/api-monitor')
                response = conn.getresponse()
                monitor = response.read()
                conn.close()
                # The standalone /api-monitor page was retired: the gateway no longer serves or injects
                # monitor.html, so the request falls through to the backend and now requires the API key.
                self.assertEqual(response.status, 401)
                self.assertNotIn(b'API Monitor', monitor)
                self.assertIn(b'/__strata_launcher/monitor.js', page)
            finally:
                gateway.stop()

    def test_live_settings_persist_without_changing_public_port(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.json'
            path.write_text(json.dumps({'port': 8080, 'exe': 'engine'}))
            self.gateway.cfg_path = path
            flags = {'context_compress_enabled': False, 'fit_max_tokens': False}
            self.gateway.update_policy(flags)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['port'], 8080)
            self.assertEqual(saved['exe'], 'engine')
            self.assertEqual(self.gateway.report()['live'], flags)
            self.assertTrue(Path(str(path)+'.context.bak').is_file())

    def test_responses_stateful_input_rejected_clearly(self):
        result = self.request('/v1/responses', {'input': 'continue', 'previous_response_id': 'x'})
        self.assertEqual(result[0], 400)
        self.assertEqual(self.backend.requests, [])

    def test_mixed_anthropic_result_stays_with_tool_call(self):
        from strata_gateway import canonical
        record = {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call', 'content': 'value'}, {'type':'text','text':'continue'}]}
        converted = canonical('/v1/messages', {'messages': [record]})
        self.assertEqual(converted[0]['role'], 'tool')


class ConnectionLifecycleTests(unittest.TestCase):
    def test_disconnect_cancels_http10_backend(self):
        import socket
        entered, closed = threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def log_message(self, *args): pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: ready\n\n')
                self.wfile.flush()
                entered.set()
                self.connection.settimeout(4)
                try:
                    if self.connection.recv(1) == b'': closed.set()
                except OSError: pass
        backend = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        gateway = ContextGateway('127.0.0.1', 0, backend.server_port,
            {'context_compress_enabled': False, 'fit_max_tokens': False})
        gateway.start()
        try:
            client = socket.create_connection(('127.0.0.1', gateway.port))
            client.sendall(('GET /hold HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n' % gateway.port).encode())
            self.assertTrue(entered.wait(2))
            client.recv(4096)
            client.close()
            self.assertTrue(closed.wait(3), 'disconnect did not close the HTTP/1.0 backend socket')
        finally:
            gateway.stop(); backend.shutdown(); backend.server_close()


if __name__ == '__main__':
    unittest.main()
