# mjviser

Web-based MuJoCo viewer powered by [viser](https://github.com/nerfstudio-project/viser).

## Install

```bash
pip install mjviser
```

## Quick start

View any MuJoCo model from the command line:

```bash
mjviser model.xml
```

Or from Python:

```python
import mujoco
from mjviser import Viewer

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
Viewer(model, data).run()
```

Open the printed URL in your browser. You get pause/resume, speed controls, single-stepping, reset, keyframe selection, joint sliders, and actuator sliders out of the box.

### Custom controller

```python
def controller(model, data):
    data.ctrl[:] = compute_ctrl(data)
    mujoco.mj_step(model, data)

Viewer(model, data, step_fn=controller).run()
```

### Passive mode

Use `ViserMujocoScene` directly if you want full control over the loop. The `server` is a standard [viser](https://viser.studio) server, so you can add custom GUI on top.

```python
server = viser.ViserServer()
scene = ViserMujocoScene.create(server, model, num_envs=1)
scene.create_visualization_gui()

while True:
    mujoco.mj_step(model, data)
    scene.update_from_mjdata(data)
```

### Multi-environment

Pass `render_fn` and `num_envs` for batched simulation (e.g. with mujoco-warp):

```python
Viewer(model, data, step_fn=step, render_fn=render, num_envs=4).run()
```

## Examples

- `active_viewer.py`: viewer with playback controls
- `active_viewer_with_controller.py`: custom step function with random torques
- `passive_viewer.py`: you own the simulation loop
- `multi_env.py`: 4 humanoids in parallel via mujoco-warp
