"""Active MuJoCo viewer with realtime pacing and playback controls."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from threading import Lock

import mujoco
import numpy as np
import viser

from .scene import ViserMujocoScene

_SPEEDS = [1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0]
_FRAME_TIME = 1.0 / 60.0  # 60 Hz render target.


def _format_speed(multiplier: float) -> str:
  if multiplier == 1.0:
    return "1x"
  inv = 1.0 / multiplier
  inv_rounded = round(inv)
  if abs(inv - inv_rounded) < 1e-9 and inv_rounded > 0:
    return f"1/{inv_rounded}x"
  return f"{multiplier:.3g}x"


class Viewer:
  """Active viewer that steps a MuJoCo simulation with realtime pacing.

  Handles the simulation loop, timing, pause/resume, speed controls,
  and scene rendering.

  For full control over the loop, use ``ViserMujocoScene`` directly
  and call ``update_from_mjdata`` or ``update_from_arrays`` yourself.
  """

  def __init__(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step_fn: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    render_fn: Callable[[ViserMujocoScene], None] | None = None,
    reset_fn: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    num_envs: int = 1,
    server: viser.ViserServer | None = None,
  ) -> None:
    """Create an active viewer.

    Args:
      model: MuJoCo model.
      data: MuJoCo data (used for keyframes, timestep, and as the
        default render/reset source when callbacks are None).
      step_fn: Called each simulation step. Signature is
        ``step_fn(model, data)``. Defaults to ``mujoco.mj_step``.
      render_fn: Called each render frame to push state to the scene.
        Signature is ``render_fn(scene)``. Defaults to
        ``scene.update_from_mjdata(data)``. Override this for
        multi-environment or custom rendering.
      reset_fn: Called on reset. Signature is
        ``reset_fn(model, data)``. Defaults to
        ``mj_resetData + mj_forward``. Override this to reset
        custom simulation state (e.g. warp data).
      num_envs: Number of parallel environments (passed to
        ``ViserMujocoScene.create``).
      server: Optional viser server. If None, one is created.
    """
    self.model = model
    self.data = data
    self._step_fn = step_fn or mujoco.mj_step
    self._render_fn = render_fn
    self._reset_fn = reset_fn
    self._server = server or viser.ViserServer()
    self.scene = ViserMujocoScene.create(self._server, model, num_envs=num_envs)
    self._lock = Lock()

    # Speed.
    self._speed_idx = _SPEEDS.index(1.0)
    self._joint_sliders: list = []

    # State.
    self._paused = False
    self._step_count = 0

    # Timing.
    self._budget = 0.0
    self._last_tick = 0.0
    self._was_capped = False

    # Render timer.
    self._time_until_next_render = 0.0

    # Windowed stats, updated every 0.5s.
    self._stats_steps = 0
    self._stats_frames = 0
    self._stats_last_time = 0.0
    self._fps = 0.0
    self._sps = 0.0

  @property
  def speed(self) -> float:
    return _SPEEDS[self._speed_idx]

  @property
  def actual_realtime(self) -> float:
    return self._sps * self.model.opt.timestep

  def _render(self) -> None:
    """Push current state to the scene."""
    if self._render_fn is not None:
      self._render_fn(self.scene)
    else:
      self.scene.update_from_mjdata(self.data)

  def _reset(self) -> None:
    """Reset the simulation."""
    if self._reset_fn is not None:
      self._reset_fn(self.model, self.data)
    else:
      mujoco.mj_resetData(self.model, self.data)
      mujoco.mj_forward(self.model, self.data)

  def run(self) -> None:
    """Run the viewer loop until Ctrl+C."""
    self._setup_gui()

    prev_handler = signal.getsignal(signal.SIGINT)
    interrupted = False

    def _on_sigint(signum, frame):
      nonlocal interrupted
      interrupted = True
      signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _on_sigint)

    if self._render_fn is None:
      mujoco.mj_forward(self.model, self.data)
    self._render()

    now = time.perf_counter()
    self._last_tick = now
    self._stats_last_time = now
    try:
      while not interrupted:
        self._tick()
        time.sleep(0.001)
    finally:
      self._server.stop()
      signal.signal(signal.SIGINT, prev_handler)

  def _tick(self) -> None:
    now = time.perf_counter()
    dt = now - self._last_tick
    self._last_tick = now

    if not self._paused:
      with self._lock:
        self._step_physics(dt)

    # Render at fixed frame rate.
    self._time_until_next_render -= dt
    if self._time_until_next_render > 0:
      return

    self._time_until_next_render += _FRAME_TIME
    if self._time_until_next_render < -_FRAME_TIME:
      self._time_until_next_render = 0.0

    self._render()
    self._stats_frames += 1
    self._update_stats()

  def _step_physics(self, dt: float) -> None:
    """Run physics steps for this frame's sim-time budget."""
    step_dt = self.model.opt.timestep
    self._budget += dt * self.speed
    self._was_capped = False

    if self._budget < step_dt:
      return

    deadline = time.perf_counter() + _FRAME_TIME
    hit_deadline = False
    while self._budget >= step_dt:
      self._step_fn(self.model, self.data)
      self._budget -= step_dt
      self._step_count += 1
      self._stats_steps += 1
      if time.perf_counter() > deadline:
        hit_deadline = True
        break

    if hit_deadline:
      self._was_capped = self._budget >= step_dt
      self._budget = min(self._budget, step_dt)

  def _update_stats(self) -> None:
    if self._paused:
      return
    now = time.perf_counter()
    dt = now - self._stats_last_time
    if dt >= 0.5:
      self._fps = self._stats_frames / dt
      self._sps = self._stats_steps / dt
      self._stats_frames = 0
      self._stats_steps = 0
      self._stats_last_time = now
      self._update_status_display()

  def _update_status_display(self) -> None:
    actual_rt = self.actual_realtime
    rt_display = f"{actual_rt:.2f}x" if actual_rt > 0 else "\u2014"
    capped = ' <span style="color:#e74c3c;">[CAPPED]</span>' if self._was_capped else ""
    self._status_html.content = f"""
      <div style="font-size: 0.85em; line-height: 1.25;
                  padding: 0 1em 0.5em 1em;">
        <strong>Status:</strong>
        {"Paused" if self._paused else "Running"}{capped}<br/>
        <strong>Steps:</strong> {self._step_count}<br/>
        <strong>Speed:</strong> {_format_speed(self.speed)}<br/>
        <strong>Target RT:</strong> {self.speed:.2f}x<br/>
        <strong>Actual RT:</strong>
        {rt_display} ({self._fps:.0f} FPS)
      </div>
      """

  def _setup_gui(self) -> None:
    tabs = self._server.gui.add_tab_group()

    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
      self._status_html = self._server.gui.add_html("")

      pause_btn = self._server.gui.add_button(
        "Pause" if not self._paused else "Play",
        icon=(viser.Icon.PLAYER_PAUSE if not self._paused else viser.Icon.PLAYER_PLAY),
      )

      @pause_btn.on_click
      def _(_) -> None:
        self._paused = not self._paused
        if not self._paused:
          self._budget = 0.0
          self._last_tick = time.perf_counter()
        pause_btn.label = "Pause" if not self._paused else "Play"
        pause_btn.icon = (
          viser.Icon.PLAYER_PAUSE if not self._paused else viser.Icon.PLAYER_PLAY
        )
        for sl in self._joint_sliders:
          sl.disabled = not self._paused
        self._update_status_display()

      step_btn = self._server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)

      @step_btn.on_click
      def _(_) -> None:
        if self._paused:
          with self._lock:
            self._step_fn(self.model, self.data)
            self._step_count += 1
            self._render()
            self._update_status_display()

      reset_btn = self._server.gui.add_button("Reset")

      @reset_btn.on_click
      def _(_) -> None:
        with self._lock:
          self._reset()
          self._step_count = 0
          self._budget = 0.0
          self._render()
          self._update_status_display()

      speed_btns = self._server.gui.add_button_group(
        "Speed", options=["Slower", "1x", "Faster"]
      )

      @speed_btns.on_click
      def _(event) -> None:
        if event.target.value == "Slower":
          self._speed_idx = max(0, self._speed_idx - 1)
        elif event.target.value == "Faster":
          self._speed_idx = min(len(_SPEEDS) - 1, self._speed_idx + 1)
        else:
          self._speed_idx = _SPEEDS.index(1.0)
        self._update_status_display()

      # Keyframe selector (only if model has keyframes).
      if self.model.nkey > 0:
        key_names = []
        for i in range(self.model.nkey):
          name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
          key_names.append(name if name else f"key_{i}")

        with self._server.gui.add_folder("Keyframes"):
          options = ["(none)", *key_names]
          key_dropdown = self._server.gui.add_dropdown("Key", options=options)

          @key_dropdown.on_update
          def _(_) -> None:
            if key_dropdown.value == "(none)":
              return
            idx = key_names.index(key_dropdown.value)
            with self._lock:
              mujoco.mj_resetDataKeyframe(self.model, self.data, idx)
              mujoco.mj_forward(self.model, self.data)
              if self._reset_fn is not None:
                self._reset_fn(self.model, self.data)
              self._step_count = 0
              self._budget = 0.0
              self._render()
              self._update_status_display()

      # Scene controls (camera, environment, contacts).
      with self._server.gui.add_folder("Scene"):
        self.scene.create_visualization_gui()

    # Groups tab (geom/site visibility).
    self.scene.create_groups_gui(tabs)

    # Actuation tab (joint/actuator sliders).
    with tabs.add_tab("Actuation", icon=viser.Icon.ADJUSTMENTS):
      self._setup_joint_sliders()
      self._setup_actuator_sliders()

  def _setup_joint_sliders(self) -> None:
    """Add per-joint sliders for hinge and slide joints."""
    # Only hinge (3) and slide (2) joints get sliders.
    joints = []
    for i in range(self.model.njnt):
      jtype = int(self.model.jnt_type[i])
      if jtype not in (2, 3):  # slide, hinge
        continue
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
      if not name:
        name = f"joint_{i}"
      limited = bool(self.model.jnt_limited[i])
      if limited:
        lo, hi = self.model.jnt_range[i]
      else:
        lo, hi = (-np.pi, np.pi) if jtype == 3 else (-1.0, 1.0)
      joints.append((i, name, round(float(lo), 3), round(float(hi), 3)))

    if not joints:
      return

    with self._server.gui.add_folder("Joints"):
      for jnt_id, name, lo, hi in joints:
        qpos_adr = int(self.model.jnt_qposadr[jnt_id])
        slider = self._server.gui.add_slider(
          name,
          min=lo,
          max=hi,
          step=round((hi - lo) / 200, 4),
          initial_value=round(float(self.data.qpos[qpos_adr]), 3),
          disabled=not self._paused,
        )
        self._joint_sliders.append(slider)

        def _on_update(_, _adr=qpos_adr, _sl=slider) -> None:
          with self._lock:
            self.data.qpos[_adr] = _sl.value
            mujoco.mj_forward(self.model, self.data)
            self._render()

        slider.on_update(_on_update)

  def _setup_actuator_sliders(self) -> None:
    """Add per-actuator control sliders."""
    if self.model.nu == 0:
      return

    actuators = []
    for i in range(self.model.nu):
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
      if not name:
        name = f"actuator_{i}"
      limited = bool(self.model.actuator_ctrllimited[i])
      if limited:
        lo, hi = self.model.actuator_ctrlrange[i]
      else:
        lo, hi = -1.0, 1.0
      actuators.append((i, name, round(float(lo), 3), round(float(hi), 3)))

    with self._server.gui.add_folder("Actuators"):
      for act_id, name, lo, hi in actuators:
        slider = self._server.gui.add_slider(
          name,
          min=lo,
          max=hi,
          step=round((hi - lo) / 200, 4),
          initial_value=round(float(self.data.ctrl[act_id]), 3),
        )

        def _on_update(_, _id=act_id, _sl=slider) -> None:
          with self._lock:
            self.data.ctrl[_id] = _sl.value

        slider.on_update(_on_update)
