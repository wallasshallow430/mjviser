"""Humanoid ragdolling under gravity with playback controls.

uv run examples/active_viewer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

from mjviser import Viewer


def main() -> None:
  if len(sys.argv) > 1:
    model = mujoco.MjModel.from_xml_path(sys.argv[1])
  else:
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "humanoid.xml"))

  data = mujoco.MjData(model)
  Viewer(model, data).run()


if __name__ == "__main__":
  main()
