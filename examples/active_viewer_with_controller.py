"""Humanoid with random torques applied each step via ``step_fn``.

uv run examples/active_viewer_with_controller.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mjviser import Viewer


def random_torques(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Apply small random torques each step."""
  data.ctrl[:] = np.random.uniform(-0.5, 0.5, size=model.nu)
  mujoco.mj_step(model, data)


def main() -> None:
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "humanoid.xml"))
  data = mujoco.MjData(model)
  Viewer(model, data, step_fn=random_torques).run()


if __name__ == "__main__":
  main()
