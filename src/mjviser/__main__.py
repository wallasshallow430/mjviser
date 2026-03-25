"""Allow ``python -m mjviser model.xml`` to launch the active viewer."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

from mjviser import Viewer


def main() -> None:
  if len(sys.argv) < 2:
    print("Usage: python -m mjviser <model.xml>")
    sys.exit(1)

  path = sys.argv[1]
  if not Path(path).exists():
    print(f"File not found: {path}")
    sys.exit(1)

  model = mujoco.MjModel.from_xml_path(path)
  data = mujoco.MjData(model)
  Viewer(model, data).run()


if __name__ == "__main__":
  main()
