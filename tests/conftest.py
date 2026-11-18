from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types


os.environ.setdefault('KIVY_NO_ARGS', '1')
os.environ.setdefault('KIVY_NO_CONSOLELOG', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp')

PACKAGE_DIR = Path(__file__).resolve().parents[1] / 'glowtracker'
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

if importlib.util.find_spec('itk') is None:
    sys.modules['itk'] = types.ModuleType('itk')
