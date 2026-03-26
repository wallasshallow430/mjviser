"""Manages all Viser visualization handles and state for MuJoCo models."""

from __future__ import annotations

from collections import defaultdict

import mujoco
import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj

from .conversions import (
  get_body_name,
  group_geoms_by_visual_compat,
  is_fixed_body,
  merge_geoms,
  merge_sites,
)

# Viser visualization defaults.
_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150
_DEFAULT_ENVIRONMENT_INTENSITY = 0.8
_DEFAULT_CONTACT_POINT_COLOR = (230, 153, 51)
_DEFAULT_CONTACT_FORCE_COLOR = (255, 0, 0)

# Map from frame mode name to mjtFrame enum value.
_FRAME_MODES: dict[str, int] = {
  "None": int(mujoco.mjtFrame.mjFRAME_NONE),
  "Body": int(mujoco.mjtFrame.mjFRAME_BODY),
  "Geom": int(mujoco.mjtFrame.mjFRAME_GEOM),
  "Site": int(mujoco.mjtFrame.mjFRAME_SITE),
}

# Geom type constants (avoid repeated int() casts).
_CYLINDER = int(mjtGeom.mjGEOM_CYLINDER)
_CAPSULE = int(mjtGeom.mjGEOM_CAPSULE)
_BOX = int(mjtGeom.mjGEOM_BOX)
_ELLIPSOID = int(mjtGeom.mjGEOM_ELLIPSOID)
_SPHERE = int(mjtGeom.mjGEOM_SPHERE)
_ARROW = int(mjtGeom.mjGEOM_ARROW)
_ARROW1 = int(mjtGeom.mjGEOM_ARROW1)
_ARROW2 = int(mjtGeom.mjGEOM_ARROW2)
_ARROWS = (_ARROW, _ARROW1, _ARROW2)

_CAT_DECOR = int(mujoco.mjtCatBit.mjCAT_DECOR)
_OBJ_TENDON = int(mujoco.mjtObj.mjOBJ_TENDON)

# Cached unit meshes for decor rendering, keyed by mjtGeom int value.
_UNIT_MESHES: dict[int, trimesh.Trimesh] = {}


def _get_unit_mesh(geom_type: int) -> trimesh.Trimesh:
  """Return a cached unit mesh for the given mjvGeom type."""
  if geom_type not in _UNIT_MESHES:
    if geom_type in (_CYLINDER, _CAPSULE):
      # Capsules use cylinders: non-uniform scaling distorts hemispherical caps.
      _UNIT_MESHES[geom_type] = trimesh.creation.cylinder(radius=1.0, height=1.0)
    elif geom_type == _BOX:
      _UNIT_MESHES[geom_type] = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    elif geom_type in (_ELLIPSOID, _SPHERE):
      _UNIT_MESHES[geom_type] = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    elif geom_type in _ARROWS:
      # Composite shaft + head along Z axis, total height = 1.
      shaft = trimesh.creation.cylinder(radius=0.4, height=0.8, sections=12)
      shaft.apply_translation([0, 0, 0.4])
      head = trimesh.creation.cone(radius=1.0, height=0.2, sections=12)
      head.apply_translation([0, 0, 0.9])
      _UNIT_MESHES[geom_type] = trimesh.util.concatenate([shaft, head])
    else:
      _UNIT_MESHES[geom_type] = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
  return _UNIT_MESHES[geom_type]


class ViserMujocoScene:
  """Manages Viser scene handles and visualization state for MuJoCo models.

  This class handles geometry creation, batched rendering of bodies and
  sites across multiple environments, contact visualization, and GUI
  controls. Users build custom overlays on top using the ``server``
  attribute and viser's native API.
  """

  def __init__(
    self,
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    num_envs: int,
  ) -> None:
    """Create and populate a scene with geometry.

    Args:
        server: Viser server instance.
        mj_model: MuJoCo model.
        num_envs: Number of parallel environments.
    """
    self.server = server
    self.mj_model = mj_model
    self.mj_data = mujoco.MjData(mj_model)
    self.num_envs = num_envs

    # mjvScene infrastructure for decor visualization.
    self._mjv_scene = mujoco.MjvScene(mj_model, maxgeom=2000)
    self._mjv_option = mujoco.MjvOption()
    self._mjv_camera = mujoco.MjvCamera()
    # Disable all vis flags; properties below enable them on demand.
    self._mjv_option.flags[:] = 0
    self._mjv_option.frame = int(mujoco.mjtFrame.mjFRAME_NONE)

    # Handles.
    self.mesh_handles_by_group: dict[tuple[int, int, int], viser.BatchedGlbHandle] = {}
    self.site_handles_by_group: dict[tuple[int, int], viser.BatchedGlbHandle] = {}
    self._decor_handles: dict[int, viser.BatchedMeshHandle] = {}
    self._fixed_geom_handles: dict[tuple[int, int, int], viser.GlbHandle] = {}
    self._fixed_site_handles: dict[tuple[int, int], viser.GlbHandle] = {}

    # Visualization settings.
    self.env_idx = 0
    self.camera_tracking_enabled = True
    self.show_only_selected = False
    self.geom_groups_visible = [True, True, True, False, False, False]
    self.site_groups_visible = [True, True, True, False, False, False]
    self.contact_point_color: tuple[int, int, int] = _DEFAULT_CONTACT_POINT_COLOR
    self.contact_force_color: tuple[int, int, int] = _DEFAULT_CONTACT_FORCE_COLOR
    self.meansize_override: float | None = None
    self.needs_update = False
    self.paused = False

    # Cached visualization state for re-rendering when settings change.
    self._tracked_body_id: int | None = None
    self._last_body_xpos: np.ndarray | None = None
    self._last_body_xmat: np.ndarray | None = None
    self._last_mocap_pos: np.ndarray | None = None
    self._last_mocap_quat: np.ndarray | None = None
    self._last_env_idx = 0
    self._last_mj_data: mujoco.MjData | None = None
    self._scene_offset = np.zeros(3)

    # Build the scene.
    server.scene.configure_environment_map(
      environment_intensity=_DEFAULT_ENVIRONMENT_INTENSITY
    )
    self.fixed_bodies_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)
    self._add_fixed_geometry()
    self._create_mesh_handles_by_group()
    self._add_fixed_sites()
    self._create_site_handles_by_group()

    for body_id in range(mj_model.nbody):
      if not is_fixed_body(mj_model, body_id):
        self._tracked_body_id = body_id
        break

  # -- Overlay visibility properties (backed by mjvOption flags) -----------

  @property
  def show_contact_points(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])

  @show_contact_points.setter
  def show_contact_points(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(value)

  @property
  def show_contact_forces(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])

  @show_contact_forces.setter
  def show_contact_forces(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(value)

  @property
  def show_tendons(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON])

  @show_tendons.setter
  def show_tendons(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = int(value)

  @property
  def show_inertia(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA])

  @show_inertia.setter
  def show_inertia(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = int(value)

  @property
  def frame_mode(self) -> str:
    current = int(self._mjv_option.frame)
    for name, val in _FRAME_MODES.items():
      if val == current:
        return name
    return "None"

  @frame_mode.setter
  def frame_mode(self, value: str) -> None:
    self._mjv_option.frame = _FRAME_MODES.get(value, 0)

  def _any_decor_visible(self) -> bool:
    """Return True if any overlay visualization is enabled."""
    return (
      self.show_contact_points
      or self.show_contact_forces
      or self.show_tendons
      or self.show_inertia
      or self.frame_mode != "None"
    )

  def _sync_visibilities(self) -> None:
    """Synchronize all handle visibilities based on current flags."""
    for (_, group_id, _), handle in self.mesh_handles_by_group.items():
      handle.visible = group_id < 6 and self.geom_groups_visible[group_id]

    for (_, group_id, _), handle in self._fixed_geom_handles.items():
      handle.visible = group_id < 6 and self.geom_groups_visible[group_id]

    for (_, group_id), handle in self.site_handles_by_group.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    for (_, group_id), handle in self._fixed_site_handles.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    if not self._any_decor_visible():
      for handle in self._decor_handles.values():
        handle.visible = False

  def create_visualization_gui(
    self,
    camera_distance: float = -1.0,
    camera_azimuth: float = 120.0,
    camera_elevation: float = 20.0,
  ) -> None:
    """Add camera, environment, and contact controls into the current
    GUI context. Call this inside a tab or folder.

    Args:
        camera_distance: Default camera distance. If negative, derived
            from model extent.
        camera_azimuth: Default camera azimuth angle in degrees.
        camera_elevation: Default camera elevation angle in degrees.
    """
    if self.num_envs > 1:
      with self.server.gui.add_folder("Environment"):
        env_slider = self.server.gui.add_slider(
          "Select",
          min=0,
          max=self.num_envs - 1,
          step=1,
          initial_value=self.env_idx,
          hint=f"Select environment (0-{self.num_envs - 1})",
        )

        @env_slider.on_update
        def _(_) -> None:
          self.env_idx = int(env_slider.value)
          self.request_update()

        show_only_cb = self.server.gui.add_checkbox(
          "Hide others",
          initial_value=self.show_only_selected,
          hint="Show only the selected environment.",
        )

        @show_only_cb.on_update
        def _(_) -> None:
          self.show_only_selected = show_only_cb.value
          self.request_update()

    _center = self.mj_model.stat.center.copy()
    _extent = self.mj_model.stat.extent
    if camera_distance <= 0:
      camera_distance = 1.5 * _extent
    _az_rad = np.deg2rad(camera_azimuth)
    _el_rad = np.deg2rad(camera_elevation)
    _camera_offset = (
      np.array(
        [
          -np.cos(_el_rad) * np.cos(_az_rad),
          -np.cos(_el_rad) * np.sin(_az_rad),
          np.sin(_el_rad),
        ]
      )
      * camera_distance
    )

    with self.server.gui.add_folder("Camera"):
      cb_camera_tracking = self.server.gui.add_checkbox(
        "Track camera",
        initial_value=self.camera_tracking_enabled,
        hint="Keep tracked body centered.",
      )

      @cb_camera_tracking.on_update
      def _(_) -> None:
        self.camera_tracking_enabled = cb_camera_tracking.value
        if self.camera_tracking_enabled:
          for client in self.server.get_clients().values():
            client.camera.position = _camera_offset
            client.camera.look_at = np.zeros(3)
        self.request_update()

      slider_fov = self.server.gui.add_slider(
        "FOV (\u00b0)",
        min=_DEFAULT_FOV_MIN,
        max=_DEFAULT_FOV_MAX,
        step=1,
        initial_value=_DEFAULT_FOV_DEGREES,
        hint="Vertical FOV in degrees.",
      )

      @slider_fov.on_update
      def _(_) -> None:
        for client in self.server.get_clients().values():
          client.camera.fov = np.radians(slider_fov.value)

      @self.server.on_client_connect
      def _(client: viser.ClientHandle) -> None:
        client.camera.fov = np.radians(slider_fov.value)
        if self.camera_tracking_enabled:
          client.camera.position = _camera_offset
          client.camera.look_at = _center

    with self.server.gui.add_folder("Contacts"):
      cb_contact_points = self.server.gui.add_checkbox(
        "Points",
        initial_value=False,
      )
      contact_point_color = self.server.gui.add_rgb(
        "Points Color", initial_value=self.contact_point_color
      )
      cb_contact_forces = self.server.gui.add_checkbox(
        "Forces",
        initial_value=False,
      )
      contact_force_color = self.server.gui.add_rgb(
        "Forces Color", initial_value=self.contact_force_color
      )

      @cb_contact_points.on_update
      def _(_) -> None:
        self.show_contact_points = cb_contact_points.value
        self._sync_visibilities()
        self.request_update()

      @contact_point_color.on_update
      def _(_) -> None:
        self.contact_point_color = contact_point_color.value
        self._clear_decor_handles()
        self.request_update()

      @cb_contact_forces.on_update
      def _(_) -> None:
        self.show_contact_forces = cb_contact_forces.value
        self._sync_visibilities()
        self.request_update()

      @contact_force_color.on_update
      def _(_) -> None:
        self.contact_force_color = contact_force_color.value
        self._clear_decor_handles()
        self.request_update()

    if self.mj_model.ntendon > 0:
      cb_tendons = self.server.gui.add_checkbox(
        "Tendons",
        initial_value=self.show_tendons,
      )

      @cb_tendons.on_update
      def _(_) -> None:
        self.show_tendons = cb_tendons.value
        self._sync_visibilities()
        self.request_update()

    with self.server.gui.add_folder("Scale"):
      _vis = self.mj_model.vis.scale
      _stat = self.mj_model.stat

      meansize_slider = self.server.gui.add_slider(
        "Global",
        min=_stat.meansize * 0.1,
        max=_stat.meansize * 5.0,
        step=_stat.meansize * 0.01,
        initial_value=_stat.meansize,
        hint="Global size for all decorations (model.stat.meansize).",
      )
      frame_len_slider = self.server.gui.add_slider(
        "Frame length",
        min=0.1,
        max=5.0,
        step=0.05,
        initial_value=_vis.framelength,
        hint="Coordinate frame axis length (vis.scale.framelength).",
      )
      frame_width_slider = self.server.gui.add_slider(
        "Frame width",
        min=0.01,
        max=1.0,
        step=0.01,
        initial_value=_vis.framewidth,
        hint="Coordinate frame axis width (vis.scale.framewidth).",
      )
      contact_slider = self.server.gui.add_slider(
        "Contact width",
        min=0.01,
        max=2.0,
        step=0.01,
        initial_value=_vis.contactwidth,
        hint="Contact point radius (vis.scale.contactwidth).",
      )
      force_slider = self.server.gui.add_slider(
        "Force width",
        min=0.01,
        max=1.0,
        step=0.01,
        initial_value=_vis.forcewidth,
        hint="Force arrow width (vis.scale.forcewidth).",
      )

      def _on_scale_update(_) -> None:  # type: ignore[no-untyped-def]
        _stat.meansize = meansize_slider.value
        _vis.framelength = frame_len_slider.value
        _vis.framewidth = frame_width_slider.value
        _vis.contactwidth = contact_slider.value
        _vis.forcewidth = force_slider.value
        self._clear_decor_handles()
        self.request_update()

      meansize_slider.on_update(_on_scale_update)
      frame_len_slider.on_update(_on_scale_update)
      frame_width_slider.on_update(_on_scale_update)
      contact_slider.on_update(_on_scale_update)
      force_slider.on_update(_on_scale_update)

  def create_groups_gui(self) -> None:
    """Add geom and site group visibility checkboxes into the
    current GUI context."""
    with self.server.gui.add_folder("Geoms"):
      for i in range(6):
        cb = self.server.gui.add_checkbox(
          f"G{i}",
          initial_value=self.geom_groups_visible[i],
          hint=f"Show/hide geometry in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          self.geom_groups_visible[group_idx] = event.target.value
          self._sync_visibilities()
          self.request_update()

    with self.server.gui.add_folder("Sites"):
      for i in range(6):
        cb = self.server.gui.add_checkbox(
          f"S{i}",
          initial_value=self.site_groups_visible[i],
          hint=f"Show/hide sites in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          self.site_groups_visible[group_idx] = event.target.value
          self._sync_visibilities()
          self.request_update()

  def update_from_arrays(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray | None = None,
    mocap_quat: np.ndarray | None = None,
    env_idx: int | None = None,
    qpos: np.ndarray | None = None,
    qvel: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
  ) -> None:
    """Update scene from batched numpy arrays.

    Args:
        body_xpos: Body positions, shape ``(num_envs, nbody, 3)``.
        body_xmat: Body rotation matrices, shape
            ``(num_envs, nbody, 3, 3)``.
        mocap_pos: Mocap body positions, shape
            ``(num_envs, nmocap, 3)``.
        mocap_quat: Mocap body quaternions (wxyz), shape
            ``(num_envs, nmocap, 4)``.
        env_idx: Environment index to visualize. If None, uses
            ``self.env_idx``.
        qpos: Joint positions. Required for contact and tendon
            visualization in the selected environment.
        qvel: Joint velocities. Required for contact and tendon
            visualization in the selected environment.
        ctrl: Controls for the selected environment.
    """
    if env_idx is None:
      env_idx = self.env_idx

    if mocap_pos is None:
      nworld = body_xpos.shape[0]
      mocap_pos = np.zeros((nworld, max(self.mj_model.nmocap, 0), 3))
    if mocap_quat is None:
      nworld = body_xpos.shape[0]
      mocap_quat = np.zeros((nworld, max(self.mj_model.nmocap, 0), 4))

    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = body_xpos[env_idx, self._tracked_body_id, :].copy()
      scene_offset = -tracked_pos

    mj_data: mujoco.MjData | None = None
    if self._any_decor_visible() and qpos is not None and qvel is not None:
      self.mj_data.qpos[:] = qpos[env_idx]
      self.mj_data.qvel[:] = qvel[env_idx]
      if ctrl is not None and self.mj_model.nu > 0:
        self.mj_data.ctrl[:] = ctrl[env_idx]
      if self.mj_model.nmocap > 0:
        self.mj_data.mocap_pos[:] = mocap_pos[env_idx]
        self.mj_data.mocap_quat[:] = mocap_quat[env_idx]
      mujoco.mj_forward(self.mj_model, self.mj_data)
      mj_data = self.mj_data

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      mj_data,
    )

  def update_from_mjdata(self, mj_data: mujoco.MjData) -> None:
    """Update scene from single-environment MuJoCo data.

    Args:
        mj_data: Single environment MuJoCo data.
    """
    body_xpos = mj_data.xpos[None, ...]
    body_xmat = mj_data.xmat.reshape(-1, 3, 3)[None, ...]
    if self.mj_model.nmocap > 0:
      mocap_pos = mj_data.mocap_pos[None, ...]
      mocap_quat = mj_data.mocap_quat[None, ...]
    else:
      mocap_pos = np.zeros((1, 0, 3))
      mocap_quat = np.zeros((1, 0, 4))
    env_idx = 0
    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = mj_data.xpos[self._tracked_body_id, :].copy()
      scene_offset = -tracked_pos

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      mj_data,
    )

  def _update_visualization(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    mj_data: mujoco.MjData | None = None,
  ) -> None:
    """Shared visualization update logic."""
    self._last_body_xpos = body_xpos
    self._last_body_xmat = body_xmat
    self._last_mocap_pos = mocap_pos
    self._last_mocap_quat = mocap_quat
    self._last_env_idx = env_idx
    self._scene_offset = scene_offset
    if mj_data is not None:
      self._last_mj_data = mj_data

    self.fixed_bodies_frame.position = scene_offset
    tile = self.show_only_selected and self.num_envs > 1
    with self.server.atomic():
      body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz
      for (body_id, _, _), handle in self.mesh_handles_by_group.items():
        if not handle.visible:
          continue
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id >= 0:
          pos, quat = self._batched_transform(
            mocap_pos, mocap_quat, mocap_id, env_idx, scene_offset, tile
          )
        else:
          pos, quat = self._batched_transform(
            body_xpos, body_xquat, body_id, env_idx, scene_offset, tile
          )
        handle.batched_positions = pos
        handle.batched_wxyzs = quat

      for (body_id, _), handle in self.site_handles_by_group.items():
        if not handle.visible:
          continue
        pos, quat = self._batched_transform(
          body_xpos, body_xquat, body_id, env_idx, scene_offset, tile
        )
        handle.batched_positions = pos
        handle.batched_wxyzs = quat

      if self._any_decor_visible() and mj_data is not None:
        self._update_decor_from_mjvscene(mj_data, scene_offset)
      elif not self._any_decor_visible():
        self._hide_all_decor()

      self.server.flush()

  def _batched_transform(
    self,
    positions: np.ndarray,
    quats: np.ndarray,
    idx: int,
    env_idx: int,
    scene_offset: np.ndarray,
    tile: bool,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (batched_positions, batched_wxyzs) for a single body."""
    if tile:
      pos = np.tile((positions[env_idx, idx] + scene_offset)[None], (self.num_envs, 1))
      quat = np.tile(quats[env_idx, idx][None], (self.num_envs, 1))
    else:
      pos = positions[..., idx, :] + scene_offset
      quat = quats[..., idx, :]
    return pos, quat

  def request_update(self) -> None:
    """Request a visualization update and trigger immediate re-render
    from cache."""
    self.needs_update = True
    self.refresh_visualization()

  def refresh_visualization(self) -> None:
    """Re-render the scene using cached visualization data."""
    if (
      self._last_body_xpos is None
      or self._last_body_xmat is None
      or self._last_mocap_pos is None
      or self._last_mocap_quat is None
    ):
      return

    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = self._last_body_xpos[
        self._last_env_idx, self._tracked_body_id, :
      ].copy()
      scene_offset = -tracked_pos

    self._update_visualization(
      self._last_body_xpos,
      self._last_body_xmat,
      self._last_mocap_pos,
      self._last_mocap_quat,
      self._last_env_idx,
      scene_offset,
      self._last_mj_data,
    )
    self.needs_update = self._any_decor_visible()

  def _add_fixed_geometry(self) -> None:
    """Add fixed world geometry to the scene."""
    # Group fixed geoms by (body_id, group_id).
    body_group_geoms: dict[tuple[int, int], list[int]] = {}
    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      if not is_fixed_body(self.mj_model, body_id):
        continue
      if self.mj_model.geom_type[i] == mjtGeom.mjGEOM_PLANE:
        body_name = get_body_name(self.mj_model, body_id)
        geom_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, i)
        self.server.scene.add_grid(
          f"/fixed_bodies/{body_name}/{geom_name}",
          infinite_grid=True,
          fade_distance=50.0,
          shadow_opacity=0.2,
          plane_opacity=0.4,
          position=self.mj_model.geom_pos[i],
          wxyz=self.mj_model.geom_quat[i],
        )
        continue
      # Skip fully transparent geoms.
      if self.mj_model.geom_rgba[i, 3] == 0:
        continue
      group_id = int(self.mj_model.geom_group[i])
      body_group_geoms.setdefault((body_id, group_id), []).append(i)

    for (body_id, group_id), geom_ids in body_group_geoms.items():
      body_name = get_body_name(self.mj_model, body_id)
      visible = group_id < 6 and self.geom_groups_visible[group_id]
      subgroups = group_geoms_by_visual_compat(self.mj_model, geom_ids)
      for sub_idx, sub_geom_ids in enumerate(subgroups):
        suffix = f"/sub{sub_idx}" if len(subgroups) > 1 else ""
        handle = self.server.scene.add_mesh_trimesh(
          f"/fixed_bodies/{body_name}/group{group_id}{suffix}",
          merge_geoms(self.mj_model, sub_geom_ids),
          cast_shadow=False,
          receive_shadow=0.2,
          position=self.mj_model.body(body_id).pos,
          wxyz=self.mj_model.body(body_id).quat,
          visible=visible,
        )
        self._fixed_geom_handles[(body_id, group_id, sub_idx)] = handle

  def _create_mesh_handles_by_group(self) -> None:
    """Create mesh handles for each geom group separately."""
    body_group_geoms: dict[tuple[int, int], list[int]] = {}

    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      if is_fixed_body(self.mj_model, body_id):
        continue
      # Skip fully transparent geoms (alpha == 0).
      if self.mj_model.geom_rgba[i, 3] == 0:
        continue
      geom_group = self.mj_model.geom_group[i]
      body_group_geoms.setdefault((body_id, geom_group), []).append(i)

    with self.server.atomic():
      for (body_id, group_id), geom_indices in body_group_geoms.items():
        body_name = get_body_name(self.mj_model, body_id)
        subgroups = group_geoms_by_visual_compat(self.mj_model, geom_indices)
        visible = group_id < 6 and self.geom_groups_visible[group_id]

        for sub_idx, sub_geom_ids in enumerate(subgroups):
          mesh = merge_geoms(self.mj_model, sub_geom_ids)
          lod_ratio = 1000.0 / mesh.vertices.shape[0]
          suffix = f"/sub{sub_idx}" if len(subgroups) > 1 else ""
          handle = self.server.scene.add_batched_meshes_trimesh(
            f"/bodies/{body_name}/group{group_id}{suffix}",
            mesh,
            batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
              self.num_envs, axis=0
            ),
            batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
              self.num_envs, axis=0
            ),
            lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
            visible=visible,
          )
          self.mesh_handles_by_group[(body_id, group_id, sub_idx)] = handle

  def _add_fixed_sites(self) -> None:
    """Add fixed site geometry to the scene as static nodes."""
    body_group_sites: dict[tuple[int, int], list[int]] = {}
    for site_id in range(self.mj_model.nsite):
      body_id = self.mj_model.site_bodyid[site_id]
      if not is_fixed_body(self.mj_model, body_id):
        continue
      group_id = int(self.mj_model.site_group[site_id])
      body_group_sites.setdefault((body_id, group_id), []).append(site_id)

    for (body_id, group_id), site_ids in body_group_sites.items():
      body_name = get_body_name(self.mj_model, body_id)
      mesh = merge_sites(self.mj_model, site_ids)
      visible = group_id < 6 and self.site_groups_visible[group_id]
      handle = self.server.scene.add_mesh_trimesh(
        f"/fixed_bodies/{body_name}/sites_group{group_id}",
        mesh,
        cast_shadow=False,
        receive_shadow=0.2,
        position=self.mj_model.body(body_id).pos,
        wxyz=self.mj_model.body(body_id).quat,
        visible=visible,
      )
      self._fixed_site_handles[(body_id, group_id)] = handle

  def _create_site_handles_by_group(self) -> None:
    """Create site handles for each site group."""
    body_group_sites: dict[tuple[int, int], list[int]] = {}
    for site_id in range(self.mj_model.nsite):
      body_id = self.mj_model.site_bodyid[site_id]
      if is_fixed_body(self.mj_model, body_id):
        continue
      group_id = int(self.mj_model.site_group[site_id])
      body_group_sites.setdefault((body_id, group_id), []).append(site_id)

    with self.server.atomic():
      for (body_id, group_id), site_ids in body_group_sites.items():
        body_name = get_body_name(self.mj_model, body_id)
        mesh = merge_sites(self.mj_model, site_ids)
        visible = group_id < 6 and self.site_groups_visible[group_id]
        handle = self.server.scene.add_batched_meshes_trimesh(
          f"/bodies/{body_name}/sites_group{group_id}",
          mesh,
          batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          lod="off",
          visible=visible,
        )
        self.site_handles_by_group[(body_id, group_id)] = handle

  def _clear_decor_handles(self) -> None:
    """Remove all decor handles and clear the cache."""
    for handle in self._decor_handles.values():
      handle.remove()
    self._decor_handles.clear()

  def _hide_all_decor(self) -> None:
    """Hide all decor handles without removing them."""
    for handle in self._decor_handles.values():
      handle.visible = False

  def _update_decor_from_mjvscene(
    self, mj_data: mujoco.MjData, scene_offset: np.ndarray
  ) -> None:
    """Update overlay visualization using mjvScene.

    Calls ``mjv_updateScene`` to generate decorative geoms (contacts,
    tendons, frames, inertia) and renders them as batched viser meshes.
    """
    mujoco.mjv_updateScene(
      self.mj_model,
      mj_data,
      self._mjv_option,
      None,
      self._mjv_camera,
      int(mujoco.mjtCatBit.mjCAT_ALL),
      self._mjv_scene,
    )

    # Group relevant geoms by type (DECOR + tendons).
    geoms_by_type: dict[int, list[int]] = defaultdict(list)
    for i in range(self._mjv_scene.ngeom):
      g = self._mjv_scene.geoms[i]
      if int(g.category) == _CAT_DECOR or int(g.objtype) == _OBJ_TENDON:
        geoms_by_type[int(g.type)].append(i)

    active_keys: set[int] = set()

    for geom_type, indices in geoms_by_type.items():
      n = len(indices)
      active_keys.add(geom_type)

      positions = np.empty((n, 3), dtype=np.float32)
      orientations = np.empty((n, 4), dtype=np.float32)
      scales = np.empty((n, 3), dtype=np.float32)
      colors = np.empty((n, 3), dtype=np.uint8)

      for j, gi in enumerate(indices):
        g = self._mjv_scene.geoms[gi]
        positions[j] = np.array(g.pos) + scene_offset
        orientations[j] = vtf.SO3.from_matrix(np.array(g.mat).reshape(3, 3)).wxyz
        size = np.array(g.size)

        # Map mjvGeom size to unit-mesh scale.
        if geom_type in (_CYLINDER, _CAPSULE):
          scales[j] = [size[0], size[0], size[2] * 2]
        elif geom_type in _ARROWS:
          scales[j] = [size[0], size[0], size[2]]
        else:
          scales[j] = size

        # Color: use user overrides for contacts/forces, MuJoCo defaults otherwise.
        rgba = np.array(g.rgba)
        color = (np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8)
        if int(g.category) == _CAT_DECOR:
          if geom_type in _ARROWS:
            color = np.array(self.contact_force_color, dtype=np.uint8)
          elif geom_type in (_CYLINDER, _CAPSULE) and self.show_contact_points:
            # DECOR cylinders with pure R/G/B are frame axes, not contacts.
            r, gc_, b = rgba[0], rgba[1], rgba[2]
            is_axis = (
              (r > 0.8 and gc_ < 0.1 and b < 0.1)
              or (r < 0.1 and gc_ > 0.8 and b < 0.1)
              or (r < 0.1 and gc_ < 0.1 and b > 0.8)
            )
            if not is_axis:
              color = np.array(self.contact_point_color, dtype=np.uint8)
        colors[j] = color

      # Create or update batched handle.
      handle = self._decor_handles.get(geom_type)
      if handle is not None and n != len(handle.batched_positions):
        handle.remove()
        handle = None
        del self._decor_handles[geom_type]

      if handle is None:
        unit = _get_unit_mesh(geom_type)
        handle = self.server.scene.add_batched_meshes_simple(
          f"/decor/{geom_type}",
          unit.vertices,
          unit.faces,
          batched_wxyzs=orientations,
          batched_positions=positions,
          batched_scales=scales,
          batched_colors=colors,
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
        self._decor_handles[geom_type] = handle
      else:
        handle.batched_positions = positions
        handle.batched_wxyzs = orientations
        handle.batched_scales = scales
        handle.visible = True

    # Hide handles for types not present this frame.
    for key, handle in self._decor_handles.items():
      if key not in active_keys:
        handle.visible = False
