"""Window geometry and responsive layout helpers for the launcher."""
from __future__ import annotations

import os
import re


GEOMETRY = re.compile(r'^(\d+)x(\d+)([+-]\d+)([+-]\d+)$')


def layout_mode(width: int) -> str:
    if int(width) < 900:
        return 'compact'
    if int(width) < 1400:
        return 'standard'
    return 'wide'


def _fit(workarea, width, height):
    left, top, right, bottom = map(int, workarea)
    available_w, available_h = max(320, right - left), max(240, bottom - top)
    width = min(max(760, int(width)), available_w)
    height = min(max(560, int(height)), max(240, available_h - 28))
    x = left + max(0, (available_w - width) // 2)
    y = top + max(0, (available_h - height) // 2)
    return width, height, x, y


def _format(values):
    width, height, x, y = values
    return f'{width}x{height}{x:+d}{y:+d}'


def initial_geometry(workarea) -> str:
    left, top, right, bottom = map(int, workarea)
    width = min(1500, max(900, round((right - left) * .82)))
    height = min(980, max(600, round((bottom - top) * .86)))
    return _format(_fit(workarea, width, height))


def restore_geometry(saved: str, workareas) -> str:
    areas = list(workareas) or [(0, 0, 1280, 720)]
    match = GEOMETRY.match(str(saved or '').strip())
    if not match:
        return initial_geometry(areas[0])
    width, height, x, y = map(int, match.groups())
    for left, top, right, bottom in areas:
        overlap_w = max(0, min(x + width, right) - max(x, left))
        overlap_h = max(0, min(y + height, bottom) - max(y, top))
        if overlap_w >= 80 and overlap_h >= 60 and width <= right - left and height <= bottom - top:
            return saved
    return _format(_fit(areas[0], width, height))


def monitor_workareas(window=None):
    """Return Windows work areas, with a Tk screen-size fallback."""
    if os.name == 'nt':
        try:
            import ctypes
            from ctypes import wintypes
            class RECT(ctypes.Structure):
                _fields_ = [('left', wintypes.LONG), ('top', wintypes.LONG),
                            ('right', wintypes.LONG), ('bottom', wintypes.LONG)]
            class MONITORINFO(ctypes.Structure):
                _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', RECT),
                            ('rcWork', RECT), ('dwFlags', wintypes.DWORD)]
            areas = []
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR,
                                               wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM)
            def collect(handle, _dc, _rect, _data):
                info = MONITORINFO(); info.cbSize = ctypes.sizeof(info)
                if ctypes.windll.user32.GetMonitorInfoW(handle, ctypes.byref(info)):
                    work = info.rcWork
                    areas.append((work.left, work.top, work.right, work.bottom))
                return True
            ctypes.windll.user32.EnumDisplayMonitors(None, None, callback_type(collect), 0)
            if areas:
                return areas
        except (AttributeError, OSError, ValueError):
            pass
    width = int(window.winfo_screenwidth()) if window else 1280
    height = int(window.winfo_screenheight()) if window else 720
    return [(0, 0, width, height)]
