"""Entry point of the public demo (Streamlit Community Cloud: main file ``demo/streamlit_app.py``).

It runs the normal dashboard (``predmaint.dashboard.app``) on the static snapshot next to this
file. The project itself is not installed there, only ``demo/requirements.txt``, so ``src`` is put
on the import path. ``runpy`` executes the app afresh on every Streamlit rerun; a plain import
would run only once and leave the page blank afterwards.
"""

import os
import runpy
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
SRC = str(DEMO.parent / "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)
if not os.environ.get("PREDMAINT_DEMO_SNAPSHOT"):
    os.environ["PREDMAINT_DEMO_SNAPSHOT"] = str(DEMO / "snapshot")

runpy.run_module("predmaint.dashboard.app", run_name="__main__")
