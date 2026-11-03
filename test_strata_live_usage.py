"""Live accounting regression: counts must change before SSE reaches EOF."""
import http.client
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from strata_gateway import ContextGateway, GatewayError
from strata_usage import UsageTracker


class LiveTrackerTests(unittest.TestCase):
    def test_progress_is_visible_without_committing_and_finishes_once(self):
        tracker = UsageTracker()
        tracker.begin('one', '/v1/messages')
        tracker.mark_forwarded('one')
        tracker.progress('one', {'input_tokens': 20, 'output_tokens': 5}, source='engine')
        snapshot = tracker.snapshot(100)
        self.assertEqual(snapshot['current_request']['output_tokens'], 5)
        self.assertEqual(snapshot['display_totals']['total_tokens'], 25)
        self.assertEqual(snapshot['totals']['total_tokens'], 0)
        self.assertEqual(snapshot['context']['used_tokens'], 25)
        tracker.progress('one', {'input_tokens': 20, 'output_tokens': 9}, source='engine')
        self.assertEqual(tracker.snapshot()['current_request']['output_tokens'], 9)
        tracker.finish('one', {'input_tokens': 20, 'output_tokens': 12}, 200, True)
        tracker.progress('one', {'input_tokens': 20, 'output_tokens': 50}, source='engine')
        final = tracker.snapshot()
        self.assertEqual(final['display_totals']['total_tokens'], 32)
        self.assertIsNone(final['current_request'])

    def test_parallel_progress_reset_and_summary_do_not_leak_into_user_totals(self):
        tracker = UsageTracker()
        for key in ('a', 'b'):
            tracker.begin(key, '/v1/messages')
            tracker.mark_forwarded(key)
            tracker.progress(key, {'input_tokens': 10, 'output_tokens': 2}, source='stream')
        tracker.record_internal_summary({'input_tokens': 100, 'output_tokens': 20})
        self.assertEqual(tracker.snapshot()['display_totals']['total_tokens'], 24)
        tracker.reset()
        tracker.progress('a', {'input_tokens': 10, 'output_tokens': 99}, source='engine')
        tracker.finish('b', {'input_tokens': 10, 'output_tokens': 5}, 200, True)
        self.assertEqual(tracker.snapshot()['display_totals']['total_tokens'], 0)

    def test_cancelled_stream_preserves_observed_counts(self):
        tracker = UsageTracker()
        tracker.begin('a', '/v1/messages')
        tracker.progress('a', {'input_tokens': 20, 'output_tokens': 3}, source='engine')
        tracker.finish('a', {}, 499, False)
        snap = tracker.snapshot()
        self.assertEqual(snap['totals']['total_tokens'], 23)
        self.assertEqual(snap['requests']['failed'], 1)

    def test_partial_input_only_usage_preserves_live_output_on_disconnect(self):
        tracker = UsageTracker()
        tracker.begin('a', '/v1/messages')
        tracker.progress('a', {'input_tokens': 20, 'output_tokens': 3}, source='engine')
        tracker.finish('a', {'input_tokens': 20}, 200, False)
        snap = tracker.snapshot()
        self.assertEqual(snap['totals']['output_tokens'], 3)
        self.assertFalse(snap['last_request']['precise'])

    def test_input_only_stream_record_is_not_final_accurate_usage(self):
        from strata_gateway import UsageStreamObserver
        observer = UsageStreamObserver()
        observer.feed(b'data: {"type":"message_start","message":{"usage":{"input_tokens":20,"output_tokens":0}}}\n\n')
        self.assertFalse(observer.finish()[1])


class LiveGatewayTests(unittest.TestCase):
    def test_metrics_finishing_race_and_reset_cannot_double_count(self):
        gateway = ContextGateway('127.0.0.1', 0, 1, {})
        gateway.context_window = 100
        gateway.usage.begin('a', '/v1/messages')
        gateway.usage.mark_forwarded('a')
        def metrics(path, **kwargs):
            self.assertEqual(kwargs['timeout'], 2)
            gateway.usage.finish('a', {'input_tokens': 20, 'output_tokens': 12}, 200, True)
            return {'live': {'state': 'generating', 'prompt_tokens': 20, 'generated': 10}}
        gateway.json_backend = metrics
        try:
            result = gateway.usage_snapshot()
            self.assertEqual(result['display_totals']['total_tokens'], 32)
            self.assertIsNone(result['current_request'])
        finally:
            gateway.stop()

    def test_unavailable_metrics_keeps_partial_stream_usage(self):
        gateway = ContextGateway('127.0.0.1', 0, 1, {})
        gateway.context_window = 100
        gateway.usage.begin('a', '/v1/messages')
        gateway.usage.mark_forwarded('a')
        gateway.usage.progress('a', {'input_tokens': 20, 'output_tokens': 2})
        def unavailable(*args, **kwargs):
            raise GatewayError('metrics unavailable')
        gateway.json_backend = unavailable
        try:
            result = gateway.usage_snapshot()
            self.assertEqual(result['display_totals']['total_tokens'], 22)
            self.assertEqual(result['current_request']['source'], 'stream')
        finally:
            gateway.stop()

    def test_summary_preparation_and_parallel_work_are_not_misattributed(self):
        gateway = ContextGateway('127.0.0.1', 0, 1, {})
        gateway.context_window = 100
        gateway.usage.begin('preparing', '/v1/messages')
        gateway.json_backend = lambda *args, **kwargs: self.fail('must not sample internal summary')
        try:
            self.assertEqual(gateway.usage_snapshot()['display_totals']['total_tokens'], 0)
            gateway.usage.mark_forwarded('preparing')
            gateway.usage.begin('parallel', '/v1/messages')
            gateway.usage.mark_forwarded('parallel')
            gateway.usage.progress('preparing', {'input_tokens': 10, 'output_tokens': 2})
            gateway.usage.progress('parallel', {'input_tokens': 20, 'output_tokens': 3})
            self.assertEqual(gateway.usage_snapshot()['display_totals']['total_tokens'], 35)
        finally:
            gateway.stop()

    def test_engine_counts_update_while_stream_is_open(self):
        started, release = threading.Event(), threading.Event()
        output = [5]
        class Backend(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def log_message(self, *args):
                pass
            def reply(self, value):
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_GET(self):
                if self.path == '/props':
                    return self.reply({'default_generation_settings': {'n_ctx': 100}})
                return self.reply({'live': {'state': 'generating', 'prompt_tokens': 20,
                                           'generated': output[0], 'running': 1}})
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: {"message":{"usage":{"input_tokens":20}}}\n\n')
                self.wfile.flush()
                started.set()
                release.wait(5)
                self.wfile.write(b'data: {"usage":{"output_tokens":12}}\n\ndata: [DONE]\n\n')
                self.wfile.flush()
        backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        gateway = ContextGateway('127.0.0.1', 0, backend.server_port,
            {'context_compress_enabled': False, 'fit_max_tokens': False}, api_key='test-key')
        gateway.start()
        errors = []
        def chat():
            conn = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=10)
            try:
                conn.request('POST', '/v1/messages', json.dumps({'stream': True, 'messages': []}),
                             {'Authorization': 'Bearer test-key', 'Content-Type': 'application/json'})
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
            except Exception as error:
                errors.append(error)
            finally:
                conn.close()
        worker = threading.Thread(target=chat)
        worker.start()
        def snapshot():
            conn = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=5)
            try:
                conn.request('GET', '/__strata_launcher/usage', headers={'Authorization': 'Bearer test-key'})
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                return json.loads(response.read())
            finally:
                conn.close()
        try:
            self.assertTrue(started.wait(3))
            first = snapshot()
            self.assertEqual(first['current_request']['output_tokens'], 5)
            self.assertEqual(first['requests']['active'], 1)
            output[0] = 9
            second = snapshot()
            self.assertEqual(second['current_request']['output_tokens'], 9)
            self.assertEqual(second['display_totals']['total_tokens'], 29)
            self.assertEqual(second['totals']['total_tokens'], 0)
            release.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertFalse(errors, errors)
            final = snapshot()
            self.assertEqual(final['totals']['total_tokens'], 32)
            self.assertIsNone(final['current_request'])
        finally:
            release.set()
            worker.join(3)
            gateway.stop()
            backend.shutdown()
            backend.server_close()
