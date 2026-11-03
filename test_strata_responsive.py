import unittest
from pathlib import Path

from strata_responsive import initial_geometry, layout_mode, restore_geometry


class ResponsiveGeometryTests(unittest.TestCase):
    def test_layout_breakpoints(self):
        self.assertEqual(layout_mode(899), 'compact')
        self.assertEqual(layout_mode(900), 'standard')
        self.assertEqual(layout_mode(1399), 'standard')
        self.assertEqual(layout_mode(1400), 'wide')

    def test_initial_geometry_fits_small_workarea(self):
        geometry = initial_geometry((0, 0, 1280, 680))
        width, rest = geometry.split('x', 1)
        height = rest.split('+', 1)[0]
        self.assertLessEqual(int(width), 1280)
        self.assertLessEqual(int(height), 680)

    def test_geometry_from_removed_monitor_moves_inside_primary_workarea(self):
        self.assertEqual(restore_geometry('1180x820+3000+100', [(0, 0, 1366, 728)]),
                         '1180x700+93+14')

    def test_negative_coordinate_monitor_remains_valid(self):
        saved = '1000x700-1500+20'
        self.assertEqual(restore_geometry(saved, [(-1920, 0, 0, 1080), (0, 0, 1920, 1040)]), saved)

    def test_broken_geometry_uses_initial_size(self):
        restored = restore_geometry('broken', [(0, 0, 1920, 1040)])
        self.assertRegex(restored, r'^\d+x\d+\+\d+\+\d+$')

    def test_launcher_dialogs_do_not_use_fixed_geometry(self):
        source = Path('strata_launcher.py').read_text(encoding='utf-8-sig')
        self.assertNotIn('geometry("760x440")', source)
        self.assertNotIn("geometry('860x560')", source)


if __name__ == '__main__':
    unittest.main()
