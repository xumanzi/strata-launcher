import threading
import unittest

from strata_usage import UsageTracker, normalize_usage


class UsageNormalizationTests(unittest.TestCase):
    def test_normalizes_openai_and_responses_shapes(self):
        self.assertEqual(normalize_usage({'prompt_tokens': 10, 'completion_tokens': 4})[:3], (10, 4, 14))
        self.assertEqual(normalize_usage({'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10})[:3],
                         (7, 3, 10))

    def test_invalid_or_negative_counts_are_clamped(self):
        self.assertEqual(normalize_usage({'input_tokens': -2, 'output_tokens': 'bad'})[:3], (0, 0, 0))


class UsageTrackerTests(unittest.TestCase):
    def test_internal_summary_is_separate_from_user_totals(self):
        tracker = UsageTracker()
        tracker.begin('user-1', '/v1/chat/completions', 'm')
        tracker.finish('user-1', {'prompt_tokens': 20, 'completion_tokens': 5}, 200, True)
        tracker.record_internal_summary({'prompt_tokens': 8, 'completion_tokens': 2})
        snap = tracker.snapshot(32768)
        self.assertEqual(snap['totals'], {'input_tokens': 20, 'output_tokens': 5, 'total_tokens': 25})
        self.assertEqual(snap['summary_usage']['total_tokens'], 10)
        self.assertEqual(snap['requests']['successful'], 1)

    def test_failure_and_compression_are_reported(self):
        tracker = UsageTracker()
        tracker.begin('bad', '/v1/messages', 'm')
        tracker.finish('bad', {}, 500, False)
        tracker.compression({'prompt_before': 120, 'prompt_after': 70, 'summarized_messages': 6}, cached=True)
        snap = tracker.snapshot(256)
        self.assertEqual(snap['requests']['failed'], 1)
        self.assertEqual(snap['compression']['count'], 1)
        self.assertEqual(snap['compression']['saved_tokens'], 50)
        self.assertEqual(snap['compression']['cache_hits'], 1)

    def test_reset_during_concurrent_finishes_keeps_consistent_nonnegative_snapshot(self):
        tracker = UsageTracker()

        def worker(index):
            key = f'r-{index}'
            tracker.begin(key, '/v1/messages', 'm')
            tracker.finish(key, {'input_tokens': 3, 'output_tokens': 2}, 200, True)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
        for thread in threads:
            thread.start()
        tracker.reset()
        for thread in threads:
            thread.join()
        snap = tracker.snapshot(100)
        self.assertTrue(all(value >= 0 for value in snap['totals'].values()))
        self.assertEqual(snap['totals']['total_tokens'],
                         snap['totals']['input_tokens'] + snap['totals']['output_tokens'])
        self.assertGreaterEqual(snap['requests']['active'], 0)


if __name__ == '__main__':
    unittest.main()
