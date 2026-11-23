import unittest

import strata_launcher as launcher


class FriendlyLabelTests(unittest.TestCase):
    def test_runtime_modes_use_plain_language_labels(self):
        self.assertEqual(launcher.RUNTIME_MODE_LABELS, {
            'speed': '速度优先（已调优）',
            'single': '单任务优先（推荐）',
            'concurrent': '多任务同时运行',
            'session': '长对话快速恢复（高级）',
        })

    def test_acceleration_labels_keep_exact_engine_values(self):
        self.assertEqual(launcher.ACCELERATION_LABELS, {
            'original': '自动（推荐）',
            '4': '标准加速',
            '6': '增强加速（实验）',
        })

    def test_thinking_labels_are_beginner_friendly(self):
        self.assertEqual(launcher.THINKING_LABELS, {
            'none': '关闭思考（最快）',
            'high': '深入思考',
            'medium': '均衡思考',
            'low': '快速回答',
        })
        self.assertEqual(launcher.THINKING_BUDGET_LABELS[0], '由模型自动决定（推荐）')

    def test_friendly_choices_round_trip_to_engine_settings(self):
        options = launcher.optimization_from_labels(
            '深入思考', '长对话快速恢复（高级）',
            '由模型自动决定（推荐）', '增强加速（实验）', True)
        self.assertEqual(options, {
            'effort': 'high', 'mode': 'session', 'budget': 0,
            'spec': '6', 'profile': True,
        })


if __name__ == '__main__':
    unittest.main()
