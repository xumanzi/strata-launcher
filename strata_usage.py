"""Thread-safe, content-free usage accounting for the Strata gateway."""
from __future__ import annotations

import copy
import threading
import time


def _count(value) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def normalize_usage(value) -> tuple[int, int, int, bool]:
    """Return input, output, total and whether a backend usage object existed."""
    value = value if isinstance(value, dict) else {}
    input_n = _count(value.get('prompt_tokens', value.get('input_tokens')))
    output_n = _count(value.get('completion_tokens', value.get('output_tokens')))
    supplied_total = value.get('total_tokens')
    total_n = _count(supplied_total) if supplied_total is not None else input_n + output_n
    total_n = max(total_n, input_n + output_n)
    return input_n, output_n, total_n, bool(value)


class UsageTracker:
    """Accumulate counts without retaining prompts or model output."""

    def __init__(self):
        self._lock = threading.RLock()
        self._generation = 0
        self.reset()

    def reset(self):
        with self._lock:
            self._generation += 1
            self._started_at = time.time()
            self._active = {}
            self._totals = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            self._requests = {'total': 0, 'successful': 0, 'failed': 0, 'active': 0}
            self._last = None
            self._summary = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'requests': 0}
            self._compression = {'count': 0, 'cache_hits': 0, 'saved_tokens': 0, 'last': None}

    def begin(self, request_id: str, path: str, model: str = ''):
        with self._lock:
            self._active[request_id] = {
                'generation': self._generation, 'path': path, 'model': model,
                'started_at': time.time(),
            }
            self._requests['total'] += 1
            self._requests['active'] = len(self._active)

    def mark_forwarded(self, request_id):
        with self._lock:
            if request_id in self._active:
                self._active[request_id]['forwarded'] = True

    def forwarded_requests(self):
        with self._lock:
            return tuple(key for key, value in self._active.items() if value.get('forwarded'))

    def progress(self, request_id, usage, source='stream'):
        """Attach counts to a still-active request; late samples cannot recreate it."""
        input_n, output_n, _total_n, present = normalize_usage(usage)
        if not present:
            return
        with self._lock:
            request = self._active.get(request_id)
            if not request:
                return
            previous = request.get('progress') or {}
            input_n = max(input_n, previous.get('input_tokens', 0))
            output_n = max(output_n, previous.get('output_tokens', 0))
            request['progress'] = {'input_tokens': input_n, 'output_tokens': output_n,
                'total_tokens': input_n + output_n, 'precise': True,
                'source': source, 'updated_at': time.time()}

    def finish(self, request_id: str, usage, status: int, precise: bool):
        with self._lock:
            request = self._active.pop(request_id, None)
            self._requests['active'] = len(self._active)
            if not request or request['generation'] != self._generation:
                return
            input_n, output_n, total_n, present = normalize_usage(usage)
            if not precise and request.get('progress'):
                observed = request['progress']
                input_n = max(input_n, observed['input_tokens'])
                output_n = max(output_n, observed['output_tokens'])
                total_n, present = input_n + output_n, True
                precise = False  # no final usage record, e.g. interrupted stream
            success = 200 <= int(status or 0) < 400
            self._requests['successful' if success else 'failed'] += 1
            if present:
                self._totals['input_tokens'] += input_n
                self._totals['output_tokens'] += output_n
                self._totals['total_tokens'] = self._totals['input_tokens'] + self._totals['output_tokens']
            self._last = {
                'path': request['path'], 'model': request['model'], 'status': int(status or 0),
                'input_tokens': input_n, 'output_tokens': output_n,
                'total_tokens': input_n + output_n, 'precise': bool(precise and present),
                'finished_at': time.time(),
            }

    def record_internal_summary(self, usage):
        input_n, output_n, _total_n, present = normalize_usage(usage)
        if not present:
            return
        with self._lock:
            self._summary['input_tokens'] += input_n
            self._summary['output_tokens'] += output_n
            self._summary['total_tokens'] = self._summary['input_tokens'] + self._summary['output_tokens']
            self._summary['requests'] += 1

    def compression(self, stats, cached=False):
        if not isinstance(stats, dict) or not stats.get('summarized_messages'):
            return
        before = _count(stats.get('prompt_before'))
        after = _count(stats.get('prompt_after'))
        saved = max(0, before - after)
        with self._lock:
            self._compression['count'] += 1
            self._compression['cache_hits'] += int(bool(cached or stats.get('summary_cached')))
            self._compression['saved_tokens'] += saved
            self._compression['last'] = {
                'before_tokens': before, 'after_tokens': after, 'saved_tokens': saved,
                'summarized_messages': _count(stats.get('summarized_messages')),
                'cached': bool(cached or stats.get('summary_cached')), 'at': time.time(),
            }

    def snapshot(self, context_limit=0, live=None):
        with self._lock:
            result = {
                'started_at': self._started_at,
                'updated_at': time.time(),
                'context': {'used_tokens': 0, 'limit_tokens': _count(context_limit), 'percent': 0.0},
                'last_request': copy.deepcopy(self._last),
                'totals': copy.deepcopy(self._totals),
                'requests': copy.deepcopy(self._requests),
                'compression': copy.deepcopy(self._compression),
                'summary_usage': copy.deepcopy(self._summary),
            }
            active = list(self._active.values())
            result['current_request'] = None
            result['display_totals'] = dict(result['totals'])
            for request in active:
                progress = request.get('progress') or {}
                for key in ('input_tokens', 'output_tokens', 'total_tokens'):
                    result['display_totals'][key] += progress.get(key, 0)
            if active:
                request = active[-1]
                progress = request.get('progress') or {
                    'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                    'precise': False, 'source': 'waiting'}
                result['current_request'] = {**copy.deepcopy(progress),
                    'path': request['path'], 'model': request['model'], 'in_progress': True,
                    'started_at': request['started_at'], 'parallel_requests': len(active)}
        if isinstance(live, dict):
            used = _count(live.get('used_tokens', live.get('prompt_tokens', 0))) + _count(live.get('generated', 0))
        elif result['current_request']:
            used = result['current_request']['total_tokens']
        elif result['last_request']:
            used = result['last_request']['total_tokens']
        else:
            used = 0
        limit = result['context']['limit_tokens']
        result['context']['used_tokens'] = used
        result['context']['percent'] = round(100.0 * used / limit, 1) if limit else 0.0
        return result
