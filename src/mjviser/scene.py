"""Manages all Viser visualization handles and state for MuJoCo models."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj

from .conversions import (
  get_body_name,
  group_geoms_by_visual_compat,
  is_fixed_body,
  merge_geoms,
  merge_sites,
  rotation_matrix_from_vectors,
)

# Viser visualization defaults.
_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150
_DEFAULT_ENVIRONMENT_INTENSITY = 0.8
_DEFAULT_CONTACT_POINT_COLOR = (230, 153, 51)
_DEFAULT_CONTACT_FORCE_COLOR = (255, 0, 0)


@dataclass
class _Contact:
  """Contact data from MuJoCo."""

  pos: np.ndarray
  frame: np.ndarray  # 3x3 rotation matrix.
  force: np.ndarray  # Force in contact frame.
  dist: float
  included: bool


@dataclass
class _ContactPointVisual:
  """Visual representation data for a contact point."""

  position: np.ndarray
  orientation: np.ndarray  # Quaternion (wxyz).
  scale: np.ndarray  # [width, width, height].


@dataclass
class _ContactForceVisual:
  """Visual representation data for a contact force arrow."""

  shaft_position: np.ndarray
  shaft_orientation: np.ndarray  # Quaternion (wxyz).
  shaft_scale: np.ndarray  # [width, width, length].
  head_position: np.ndarray
  head_orientation: np.ndarray  # Quaternion (wxyz).
  head_scale: np.ndarray  # [width, width, width].


@dataclass
class ViserMujocoScene:
  """Manages Viser scene handles and visualization state for MuJoCo models.

  This class handles geometry creation, batched rendering of bodies and
  sites across multiple environments, contact visualization, and GUI
  controls. Users build custom overlays on top using the ``server``
  attribute and viser's native API.
  """

  # Core.
  server: viser.ViserServer
  mj_model: mujoco.MjModel
  mj_data: mujoco.MjData
  num_envs: int

  # Handles (created once).
  fixed_bodies_frame: viser.SceneNodeHandle = field(init=False)
  mesh_handles_by_group: dict[tuple[int, int, int], viser.BatchedGlbHandle] = field(
    default_factory=dict
  )
  site_handles_by_group: dict[tuple[int, int], viser.BatchedGlbHandle] = field(
    default_factory=dict
  )
  contact_point_handle: viser.BatchedMeshHandle | None = None
  contact_force_shaft_handle: viser.BatchedMeshHandle | None = None
  contact_force_head_handle: viser.BatchedMeshHandle | None = None

  # Visualization settings.
  env_idx: int = 0
  camera_tracking_enabled: bool = True
  show_only_selected: bool = False
  geom_groups_visible: list[bool] = field(
    default_factory=lambda: [True, True, True, False, False, False]
  )
  site_groups_visible: list[bool] = field(
    default_factory=lambda: [True, True, True, False, False, False]
  )
  show_contact_points: bool = False
  show_contact_forces: bool = False
  contact_point_color: tuple[int, int, int] = _DEFAULT_CONTACT_POINT_COLOR
  contact_force_color: tuple[int, int, int] = _DEFAULT_CONTACT_FORCE_COLOR
  meansize_override: float | None = None
  needs_update: bool = False
  paused: bool = False
  _tracked_body_id: int | None = field(init=False, default=None)

  # Cached visualization state for re-rendering when settings change.
  _last_body_xpos: np.ndarray | None = None
  _last_body_xmat: np.ndarray | None = None
  _last_mocap_pos: np.ndarray | None = None
  _last_mocap_quat: np.ndarray | None = None
  _last_env_idx: int = 0
  _last_contacts: list[_Contact] | None = None

  # Scene offset from camera tracking, readable by subclasses.
  _scene_offset: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)

  _fixed_site_handles: dict[tuple[int, int], viser.GlbHandle] = field(
    default_factory=dict, init=False
  )

  @staticmethod
  def create(
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    num_envs: int,
  ) -> ViserMujocoScene:
    """Create and populate scene with geometry.

    Args:
        server: Viser server instance.
        mj_model: MuJoCo model.
        num_envs: Number of parallel environments.

    Returns:
        ViserMujocoScene instance with scene populated.
    """
    mj_data = mujoco.MjData(mj_model)

    scene = ViserMujocoScene(
      server=server,
      mj_model=mj_model,
      mj_data=mj_data,
      num_envs=num_envs,
    )

    # Configure environment lighting.
    server.scene.configure_environment_map(
      environment_intensity=_DEFAULT_ENVIRONMENT_INTENSITY
    )

    # Create frame for fixed world geometry.
    scene.fixed_bodies_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)

    # Add fixed geometry (planes, terrain, etc.).
    scene._add_fixed_geometry()

    # Create mesh handles per geom group.
    scene._create_mesh_handles_by_group()

    # Add fixed site geometry.
    scene._add_fixed_sites()

    # Create site handles per site group.
    scene._create_site_handles_by_group()

    # Find first non-fixed body for camera tracking.
    for body_id in range(mj_model.nbody):
      if not is_fixed_body(mj_model, body_id):
        scene._tracked_body_id = body_id
        break

    return scene

  def _is_collision_geom(self, geom_id: int) -> bool:
    """Check if a geom is a collision geom."""
    return (
      self.mj_model.geom_contype[geom_id] != 0
      or self.mj_model.geom_conaffinity[geom_id] != 0
    )

  def _sync_visibilities(self) -> None:
    """Synchronize all handle visibilities based on current flags."""
    for (
      _body_id,
      group_id,
      _sub_idx,
    ), handle in self.mesh_handles_by_group.items():
      handle.visible = group_id < 6 and self.geom_groups_visible[group_id]

    for (_body_id, group_id), handle in self.site_handles_by_group.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    for (_body_id, group_id), handle in self._fixed_site_handles.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    if self.contact_point_handle is not None and not self.show_contact_points:
      self.contact_point_handle.visible = False

    if not self.show_contact_forces:
      if self.contact_force_shaft_handle is not None:
        self.contact_force_shaft_handle.visible = False
      if self.contact_force_head_handle is not None:
        self.contact_force_head_handle.visible = False

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
          self._request_update()

        show_only_cb = self.server.gui.add_checkbox(
          "Hide others",
          initial_value=self.show_only_selected,
          hint="Show only the selected environment.",
        )

        @show_only_cb.on_update
        def _(_) -> None:
          self.show_only_selected = show_only_cb.value
          self._request_update()

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
        self._request_update()

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
      meansize_input = self.server.gui.add_number(
        "Scale",
        step=self.mj_model.stat.meansize * 0.01,
        initial_value=self.mj_model.stat.meansize,
      )

      @cb_contact_points.on_update
      def _(_) -> None:
        self.show_contact_points = cb_contact_points.value
        self._sync_visibilities()
        self._request_update()

      @contact_point_color.on_update
      def _(_) -> None:
        self.contact_point_color = contact_point_color.value
        if self.contact_point_handle is not None:
          self.contact_point_handle.remove()
          self.contact_point_handle = None
        self._request_update()

      @cb_contact_forces.on_update
      def _(_) -> None:
        self.show_contact_forces = cb_contact_forces.value
        self._sync_visibilities()
        self._request_update()

      @contact_force_color.on_update
      def _(_) -> None:
        self.contact_force_color = contact_force_color.value
        if self.contact_force_shaft_handle is not None:
          self.contact_force_shaft_handle.remove()
          self.contact_force_shaft_handle = None
        if self.contact_force_head_handle is not None:
          self.contact_force_head_handle.remove()
          self.contact_force_head_handle = None
        self._request_update()

      @meansize_input.on_update
      def _(_) -> None:
        self.meansize_override = meansize_input.value
        self._request_update()

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
          self._request_update()

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
          self._request_update()

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
        qpos: Joint positions for contact computation.
        qvel: Joint velocities for contact computation.
        ctrl: Controls for contact computation.
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

    contacts = None
    if self.show_contact_points or self.show_contact_forces:
      if qpos is not None and qvel is not None:
        self.mj_data.qpos[:] = qpos[env_idx]
        self.mj_data.qvel[:] = qvel[env_idx]
        if ctrl is not None and self.mj_model.nu > 0:
          self.mj_data.ctrl[:] = ctrl[env_idx]
        if self.mj_model.nmocap > 0:
          self.mj_data.mocap_pos[:] = mocap_pos[env_idx]
          self.mj_data.mocap_quat[:] = mocap_quat[env_idx]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        contacts = self._extract_contacts_from_mjdata(self.mj_data)

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      contacts,
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

    contacts = self._extract_contacts_from_mjdata(mj_data)

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      contacts,
    )

  def _update_visualization(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    contacts: list[_Contact] | None,
  ) -> None:
    """Shared visualization update logic."""
    self._last_body_xpos = body_xpos
    self._last_body_xmat = body_xmat
    self._last_mocap_pos = mocap_pos
    self._last_mocap_quat = mocap_quat
    self._last_env_idx = env_idx
    self._scene_offset = scene_offset
    if contacts is not None:
      self._last_contacts = contacts

    self.fixed_bodies_frame.position = scene_offset
    with self.server.atomic():
      body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz
      for (
        body_id,
        _group_id,
        _sub_idx,
      ), handle in self.mesh_handles_by_group.items():
        if not handle.visible:
          continue
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id >= 0:
          if self.show_only_selected and self.num_envs > 1:
            single_pos = mocap_pos[env_idx, mocap_id, :] + scene_offset
            single_quat = mocap_quat[env_idx, mocap_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = mocap_pos[:, mocap_id, :] + scene_offset
            handle.batched_wxyzs = mocap_quat[:, mocap_id, :]
        else:
          if self.show_only_selected and self.num_envs > 1:
            single_pos = body_xpos[env_idx, body_id, :] + scene_offset
            single_quat = body_xquat[env_idx, body_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = body_xpos[..., body_id, :] + scene_offset
            handle.batched_wxyzs = body_xquat[..., body_id, :]

      for (body_id, _group_id), handle in self.site_handles_by_group.items():
        if not handle.visible:
          continue
        if self.show_only_selected and self.num_envs > 1:
          single_pos = body_xpos[env_idx, body_id, :] + scene_offset
          single_quat = body_xquat[env_idx, body_id, :]
          handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
          handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
        else:
          handle.batched_positions = body_xpos[..., body_id, :] + scene_offset
          handle.batched_wxyzs = body_xquat[..., body_id, :]

      if contacts is not None:
        self._update_contact_visualization(contacts, scene_offset)

      self.server.flush()

  def _request_update(self) -> None:
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

    contacts = (
      self._last_contacts
      if (self.show_contact_points or self.show_contact_forces)
      else None
    )

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
      contacts,
    )
    self.needs_update = self.show_contact_points or self.show_contact_forces

  def _add_fixed_geometry(self) -> None:
    """Add fixed world geometry to the scene."""
    body_geoms_visual: dict[int, list[int]] = {}
    body_geoms_collision: dict[int, list[int]] = {}

    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      target = body_geoms_collision if self._is_collision_geom(i) else body_geoms_visual
      target.setdefault(body_id, []).append(i)

    all_bodies = set(body_geoms_visual.keys()) | set(body_geoms_collision.keys())

    for body_id in all_bodies:
      body_name = get_body_name(self.mj_model, body_id)

      if is_fixed_body(self.mj_model, body_id):
        all_geoms = []
        if body_id in body_geoms_visual:
          all_geoms.extend(body_geoms_visual[body_id])
        if body_id in body_geoms_collision:
          all_geoms.extend(body_geoms_collision[body_id])

        if not all_geoms:
          continue

        nonplane_geom_ids: list[int] = []
        for geom_id in all_geoms:
          geom_type = self.mj_model.geom_type[geom_id]
          if geom_type == mjtGeom.mjGEOM_PLANE:
            geom_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, geom_id)
            self.server.scene.add_grid(
              f"/fixed_bodies/{body_name}/{geom_name}",
              infinite_grid=True,
              fade_distance=50.0,
              shadow_opacity=0.2,
              plane_opacity=0.4,
              position=self.mj_model.geom_pos[geom_id],
              wxyz=self.mj_model.geom_quat[geom_id],
            )
          else:
            nonplane_geom_ids.append(geom_id)

        if len(nonplane_geom_ids) > 0:
          subgroups = group_geoms_by_visual_compat(self.mj_model, nonplane_geom_ids)
          for sub_idx, sub_geom_ids in enumerate(subgroups):
            suffix = f"/sub{sub_idx}" if len(subgroups) > 1 else ""
            self.server.scene.add_mesh_trimesh(
              f"/fixed_bodies/{body_name}{suffix}",
              merge_geoms(self.mj_model, sub_geom_ids),
              cast_shadow=False,
              receive_shadow=0.2,
              position=self.mj_model.body(body_id).pos,
              wxyz=self.mj_model.body(body_id).quat,
              visible=True,
            )

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
      key = (body_id, geom_group)
      if key not in body_group_geoms:
        body_group_geoms[key] = []
      body_group_geoms[key].append(i)

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

  def _extract_contacts_from_mjdata(self, mj_data: mujoco.MjData) -> list[_Contact]:
    """Extract contact data from given MuJoCo data."""

    def make_contact(i: int) -> _Contact:
      con, force = mj_data.contact[i], np.zeros(6)
      mujoco.mj_contactForce(self.mj_model, mj_data, i, force)
      return _Contact(
        pos=con.pos.copy(),
        frame=con.frame.copy().reshape(3, 3),
        force=force[:3].copy(),
        dist=con.dist,
        included=con.efc_address >= 0,
      )

    return [make_contact(i) for i in range(mj_data.ncon)]

  def _update_contact_visualization(
    self, contacts: list[_Contact], scene_offset: np.ndarray
  ) -> None:
    """Update contact point and force visualization."""
    import trimesh

    contact_points: list[_ContactPointVisual] = []
    contact_forces: list[_ContactForceVisual] = []

    meansize = self.meansize_override or self.mj_model.stat.meansize

    for contact in contacts:
      if not contact.included:
        continue

      force_world = contact.frame.T @ contact.force
      force_mag = np.linalg.norm(force_world)

      if self.show_contact_points:
        contact_points.append(
          _ContactPointVisual(
            position=contact.pos + scene_offset,
            orientation=vtf.SO3.from_matrix(
              rotation_matrix_from_vectors(np.array([0, 0, 1]), contact.frame[0, :])
            ).wxyz,
            scale=np.array(
              [
                self.mj_model.vis.scale.contactwidth * meansize,
                self.mj_model.vis.scale.contactwidth * meansize,
                self.mj_model.vis.scale.contactheight * meansize,
              ]
            ),
          )
        )

      if self.show_contact_forces and force_mag > 1e-6:
        force_dir = force_world / force_mag
        arrow_length = (
          force_mag * (self.mj_model.vis.map.force / self.mj_model.stat.meanmass)
          if self.mj_model.stat.meanmass > 0
          else force_mag
        )
        arrow_width = self.mj_model.vis.scale.forcewidth * meansize
        force_quat = vtf.SO3.from_matrix(
          rotation_matrix_from_vectors(np.array([0, 0, 1]), force_dir)
        ).wxyz

        contact_forces.append(
          _ContactForceVisual(
            shaft_position=contact.pos + scene_offset,
            shaft_orientation=force_quat,
            shaft_scale=np.array([arrow_width, arrow_width, arrow_length]),
            head_position=(contact.pos + scene_offset + force_dir * arrow_length),
            head_orientation=force_quat,
            head_scale=np.array([arrow_width, arrow_width, arrow_width]),
          )
        )

    if contact_points:
      positions = np.array([p.position for p in contact_points], dtype=np.float32)
      orientations = np.array([p.orientation for p in contact_points], dtype=np.float32)
      scales = np.array([p.scale for p in contact_points], dtype=np.float32)
      if self.contact_point_handle is None:
        mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
        self.contact_point_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/points",
          mesh.vertices,
          mesh.faces,
          batched_wxyzs=orientations,
          batched_positions=positions,
          batched_scales=scales,
          batched_colors=np.array(self.contact_point_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
      self.contact_point_handle.batched_positions = positions
      self.contact_point_handle.batched_wxyzs = orientations
      self.contact_point_handle.batched_scales = scales
      self.contact_point_handle.visible = True
    elif self.contact_point_handle is not None:
      self.contact_point_handle.visible = False

    if contact_forces:
      shaft_positions = np.array(
        [f.shaft_position for f in contact_forces], dtype=np.float32
      )
      shaft_orientations = np.array(
        [f.shaft_orientation for f in contact_forces], dtype=np.float32
      )
      shaft_scales = np.array([f.shaft_scale for f in contact_forces], dtype=np.float32)
      head_positions = np.array(
        [f.head_position for f in contact_forces], dtype=np.float32
      )
      head_orientations = np.array(
        [f.head_orientation for f in contact_forces], dtype=np.float32
      )
      head_scales = np.array([f.head_scale for f in contact_forces], dtype=np.float32)
      if self.contact_force_shaft_handle is None:
        shaft_mesh = trimesh.creation.cylinder(radius=0.4, height=1.0)
        shaft_mesh.apply_translation([0, 0, 0.5])
        self.contact_force_shaft_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/forces/shaft",
          shaft_mesh.vertices,
          shaft_mesh.faces,
          batched_wxyzs=shaft_orientations,
          batched_positions=shaft_positions,
          batched_scales=shaft_scales,
          batched_colors=np.array(self.contact_force_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
        head_mesh = trimesh.creation.cone(radius=1.0, height=1.0, sections=8)
        self.contact_force_head_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/forces/head",
          head_mesh.vertices,
          head_mesh.faces,
          batched_wxyzs=head_orientations,
          batched_positions=head_positions,
          batched_scales=head_scales,
          batched_colors=np.array(self.contact_force_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
      assert self.contact_force_shaft_handle is not None
      assert self.contact_force_head_handle is not None
      self.contact_force_shaft_handle.batched_positions = shaft_positions
      self.contact_force_shaft_handle.batched_wxyzs = shaft_orientations
      self.contact_force_shaft_handle.batched_scales = shaft_scales
      self.contact_force_shaft_handle.visible = True
      self.contact_force_head_handle.batched_positions = head_positions
      self.contact_force_head_handle.batched_wxyzs = head_orientations
      self.contact_force_head_handle.batched_scales = head_scales
      self.contact_force_head_handle.visible = True
    elif (
      self.contact_force_shaft_handle is not None
      and self.contact_force_head_handle is not None
    ):
      self.contact_force_shaft_handle.visible = (
        self.contact_force_head_handle.visible
      ) = False
