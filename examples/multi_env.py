"""Four humanoids simulated in parallel via mujoco-warp.

uv run examples/multi_env.py
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np

from mjviser import Viewer

NUM_ENVS = 4
SPACING = 2.0


def main() -> None:
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "humanoid.xml"))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  m = mjwarp.put_model(model)
  d = mjwarp.put_data(model, data, nworld=NUM_ENVS)

  # Lay out envs on a grid.
  cols = math.ceil(math.sqrt(NUM_ENVS))
  origins = np.zeros((NUM_ENVS, 3))
  for i in range(NUM_ENVS):
    row, col = divmod(i, cols)
    origins[i, 0] = col * SPACING
    origins[i, 1] = row * SPACING

  def set_origins():
    qpos = d.qpos.numpy()
    for i in range(NUM_ENVS):
      qpos[i, 0] = origins[i, 0]
      qpos[i, 1] = origins[i, 1]
    d.qpos.assign(qpos)
    mjwarp.forward(m, d)

  set_origins()

  def step(_model, _data):
    mjwarp.step(m, d)

  def render(scene):
    xpos = d.xpos.numpy()
    xmat = d.xmat.numpy().reshape(NUM_ENVS, -1, 3, 3)
    scene.update_from_arrays(
      xpos,
      xmat,
      qpos=d.qpos.numpy(),
      qvel=d.qvel.numpy(),
      ctrl=d.ctrl.numpy() if model.nu > 0 else None,
    )

  def reset(model, data):
    new_d = mjwarp.put_data(model, data, nworld=NUM_ENVS)
    d.qpos.assign(new_d.qpos.numpy())
    d.qvel.assign(new_d.qvel.numpy())
    set_origins()

  Viewer(
    model,
    data,
    step_fn=step,
    render_fn=render,
    reset_fn=reset,
    num_envs=NUM_ENVS,
  ).run()


if __name__ == "__main__":
  main()
