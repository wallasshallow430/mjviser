"""Convert MuJoCo mesh data to trimesh format with texture support."""

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj
from PIL import Image

# ------------------------------------------------------------------
# Color / texture helpers
# ------------------------------------------------------------------


def _get_texture_id(mj_model: mujoco.MjModel, matid: int) -> int:
  """Return the RGB or RGBA texture ID for a material, or -1."""
  texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
  if texid < 0:
    texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
  return texid


def _extract_texture_image(mj_model: mujoco.MjModel, texid: int) -> Image.Image | None:
  """Extract a 2D texture as a PIL Image, or None for unsupported types."""
  w = mj_model.tex_width[texid]
  h = mj_model.tex_height[texid]
  nc = mj_model.tex_nchannel[texid]
  adr = mj_model.tex_adr[texid]
  data = mj_model.tex_data[adr : adr + w * h * nc]

  # MuJoCo uses bottom-left origin; GLTF expects top-left.
  if nc == 1:
    arr = np.flipud(data.reshape(h, w))
    return Image.fromarray(arr.astype(np.uint8), mode="L")
  elif nc in (3, 4):
    arr = np.flipud(data.reshape(h, w, nc))
    return Image.fromarray(arr.astype(np.uint8))
  return None


def _resolve_flat_rgba(mj_model: mujoco.MjModel, geom_idx: int) -> np.ndarray:
  """Resolve the flat RGBA color for a geom as uint8.

  Priority: material rgba > geom rgba.
  """
  matid = mj_model.geom_matid[geom_idx]
  if matid >= 0 and matid < mj_model.nmat:
    rgba = mj_model.mat_rgba[matid]
  else:
    rgba = mj_model.geom_rgba[geom_idx]
  return (np.clip(rgba, 0, 1) * 255).astype(np.uint8)


def _apply_flat_color(mesh: trimesh.Trimesh, rgba_uint8: np.ndarray) -> None:
  """Apply a uniform RGBA color to all vertices."""
  mesh.visual = trimesh.visual.ColorVisuals(
    vertex_colors=np.tile(rgba_uint8, (len(mesh.vertices), 1))
  )


def _cubemap_vertex_colors(
  mj_model: mujoco.MjModel,
  matid: int,
  vertices: np.ndarray,
  faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  """Project a cube map texture onto mesh triangles by normal direction.

  For each triangle, finds the cube map face most aligned with its
  normal and assigns that face's average color. If the best face is
  empty, falls back to the nearest non-empty face (handles chamfered
  edges).

  Returns (new_vertices, new_faces, vertex_colors) with duplicated
  vertices so each triangle can have its own color, or None if the
  material has no cube map texture.
  """
  texid = _get_texture_id(mj_model, matid)
  if texid < 0:
    return None

  w = mj_model.tex_width[texid]
  h = mj_model.tex_height[texid]
  nc = mj_model.tex_nchannel[texid]
  if int(mj_model.tex_type[texid]) != 1 or h != w * 6:
    return None

  adr = mj_model.tex_adr[texid]
  data = mj_model.tex_data[adr : adr + w * h * nc].reshape(6, w, w, nc)

  # Average color per cube face. Empty faces stay at [0,0,0,0].
  face_colors = np.zeros((6, 4), dtype=np.uint8)
  has_color = np.zeros(6, dtype=bool)
  for i in range(6):
    mask = data[i, :, :, : min(nc, 3)].sum(axis=2) > 0
    if mask.any():
      avg = data[i][mask].mean(axis=0).astype(np.uint8)
      face_colors[i, :nc] = avg[:nc]
      if nc < 4:
        face_colors[i, 3] = 255
      has_color[i] = True

  if not has_color.any():
    return None

  # Cube map axes: +X, -X, +Y, -Y, +Z, -Z.
  axes = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.float64,
  )

  # Per-triangle normals.
  v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
  normals = np.cross(v1 - v0, v2 - v0)
  normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)

  # For each triangle, pick the best aligned face that has color.
  dots = normals @ axes.T
  ranked = np.argsort(-dots, axis=1)
  nf = len(faces)
  tri_colors = np.zeros((nf, 4), dtype=np.uint8)
  for fi in range(nf):
    for ci in ranked[fi]:
      if has_color[ci]:
        tri_colors[fi] = face_colors[ci]
        break

  # Duplicate vertices so each triangle gets its own color.
  new_verts = vertices[faces.ravel()]
  new_faces = np.arange(nf * 3).reshape(-1, 3)
  vert_colors = np.repeat(tri_colors, 3, axis=0)

  return new_verts, new_faces, vert_colors


# ------------------------------------------------------------------
# Mesh extraction
# ------------------------------------------------------------------


def _extract_mesh_data(
  mj_model: mujoco.MjModel, geom_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Extract vertices, faces, and optional UVs for a mesh geom.

  Returns (vertices, faces, uvs). UVs are None when the mesh has no
  texture coordinates. When UVs are present, vertices and faces are
  duplicated so each face-vertex gets its own UV.
  """
  mesh_id = mj_model.geom_dataid[geom_idx]
  vert_start = int(mj_model.mesh_vertadr[mesh_id])
  vert_count = int(mj_model.mesh_vertnum[mesh_id])
  face_start = int(mj_model.mesh_faceadr[mesh_id])
  face_count = int(mj_model.mesh_facenum[mesh_id])

  vertices = mj_model.mesh_vert[vert_start : vert_start + vert_count]
  faces = mj_model.mesh_face[face_start : face_start + face_count]

  texcoord_num = mj_model.mesh_texcoordnum[mesh_id]
  if texcoord_num == 0:
    return vertices, faces, None

  texcoord_adr = mj_model.mesh_texcoordadr[mesh_id]
  texcoords = mj_model.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_num]
  face_tc_idx = mj_model.mesh_facetexcoord[face_start : face_start + face_count]

  # Duplicate vertices so each face-vertex gets its own UV.
  new_verts = vertices[faces.flatten()]
  new_uvs = texcoords[face_tc_idx.flatten()]
  new_faces = np.arange(face_count * 3).reshape(-1, 3)

  return new_verts, new_faces, new_uvs


def mujoco_mesh_to_trimesh(
  mj_model: mujoco.MjModel, geom_idx: int, verbose: bool = False
) -> trimesh.Trimesh:
  """Convert a MuJoCo mesh geometry to a trimesh with visual appearance.

  Color resolution order:
    1. 2D texture with UVs (PBR material with baseColorTexture)
    2. Cube map texture projected by triangle normals
    3. Flat material color (mat_rgba)
    4. Flat geom color (geom_rgba)
  """
  vertices, faces, uvs = _extract_mesh_data(mj_model, geom_idx)
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

  matid = mj_model.geom_matid[geom_idx]

  # Path 1: mesh has UVs and material has a 2D texture.
  if uvs is not None and matid >= 0:
    texid = _get_texture_id(mj_model, matid)
    if texid >= 0:
      image = _extract_texture_image(mj_model, texid)
      if image is not None:
        rgba = mj_model.mat_rgba[matid]
        material = trimesh.visual.material.PBRMaterial(
          baseColorFactor=rgba,
          baseColorTexture=image,
          metallicFactor=0.0,
          roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        return mesh

  # Path 2: no UVs, try cube map projection.
  if uvs is None and matid >= 0:
    result = _cubemap_vertex_colors(mj_model, matid, vertices, faces)
    if result is not None:
      new_verts, new_faces, vert_colors = result
      mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
      mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vert_colors)
      return mesh

  # Path 3/4: flat color from material or geom.
  _apply_flat_color(mesh, _resolve_flat_rgba(mj_model, geom_idx))
  return mesh


def create_primitive_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Create a mesh for primitive geom types.

  Supports sphere, box, capsule, cylinder, plane, ellipsoid, and
  heightfield.
  """
  size = mj_model.geom_size[geom_id]
  geom_type = mj_model.geom_type[geom_id]
  rgba_uint8 = _resolve_flat_rgba(mj_model, geom_id)

  if geom_type == mjtGeom.mjGEOM_SPHERE:
    mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
  elif geom_type == mjtGeom.mjGEOM_BOX:
    mesh = trimesh.creation.box(extents=2.0 * size)
  elif geom_type == mjtGeom.mjGEOM_CAPSULE:
    mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_CYLINDER:
    mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_PLANE:
    plane_x = 2.0 * size[0] if size[0] > 0 else 20.0
    plane_y = 2.0 * size[1] if size[1] > 0 else 20.0
    mesh = trimesh.creation.box((plane_x, plane_y, 0.001))
  elif geom_type == mjtGeom.mjGEOM_ELLIPSOID:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.apply_scale(size)
  elif geom_type == mjtGeom.mjGEOM_HFIELD:
    return _create_heightfield_mesh(mj_model, geom_id)
  else:
    raise ValueError(f"Unsupported primitive geom type: {geom_type}")

  _apply_flat_color(mesh, rgba_uint8)
  return mesh


def _create_heightfield_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Create a colored heightfield mesh."""
  hfield_id = mj_model.geom_dataid[geom_id]
  nrow = mj_model.hfield_nrow[hfield_id]
  ncol = mj_model.hfield_ncol[hfield_id]
  sx, sy, sz, _base = mj_model.hfield_size[hfield_id]

  offset = 0
  for k in range(hfield_id):
    offset += mj_model.hfield_nrow[k] * mj_model.hfield_ncol[k]
  hfield = mj_model.hfield_data[offset : offset + nrow * ncol].reshape(nrow, ncol)

  x = np.linspace(-sx, sx, ncol)
  y = np.linspace(-sy, sy, nrow)
  xx, yy = np.meshgrid(x, y)
  zz = hfield * sz

  vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

  face_list = []
  for r in range(nrow - 1):
    for c in range(ncol - 1):
      i0 = r * ncol + c
      i1 = i0 + 1
      i2 = i0 + ncol
      i3 = i2 + 1
      face_list.append([i0, i1, i3])
      face_list.append([i0, i3, i2])
  faces = np.array(face_list, dtype=np.int64)
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

  # Color by height using HSV.
  zz_min, zz_max = zz.min(), zz.max()
  normalized = (
    (zz - zz_min) / (zz_max - zz_min) if zz_max > zz_min else np.full_like(zz, 0.5)
  )

  hue = 0.5 - normalized * 0.45
  saturation = 0.6 - normalized * 0.2
  value = 0.4 + normalized * 0.3

  c = value * saturation
  x = c * (1 - np.abs((hue * 6) % 2 - 1))
  m = value - c

  hue_sector = (hue * 6).astype(int) % 6
  r = np.zeros_like(hue)
  g = np.zeros_like(hue)
  b = np.zeros_like(hue)

  for sector, rc, gc, bc in [
    (0, c, x, 0),
    (1, x, c, 0),
    (2, 0, c, x),
    (3, 0, x, c),
    (4, x, 0, c),
    (5, c, 0, x),
  ]:
    mask = hue_sector == sector
    r[mask] = rc[mask] if isinstance(rc, np.ndarray) else rc
    g[mask] = gc[mask] if isinstance(gc, np.ndarray) else gc
    b[mask] = bc[mask] if isinstance(bc, np.ndarray) else bc

  r += m
  g += m
  b += m

  vertex_colors = np.column_stack(
    [
      (np.clip(r, 0, 1) * 255).astype(np.uint8).ravel(),
      (np.clip(g, 0, 1) * 255).astype(np.uint8).ravel(),
      (np.clip(b, 0, 1) * 255).astype(np.uint8).ravel(),
      np.full(vertices.shape[0], 255, dtype=np.uint8),
    ]
  )

  mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
  return mesh


# ------------------------------------------------------------------
# Texture / visual utilities
# ------------------------------------------------------------------


def get_geom_texture_id(mj_model: mujoco.MjModel, geom_idx: int) -> int:
  """Return the texture ID the trimesh conversion will use, or -1.

  Returns -1 when the geom is a primitive, the material has no
  texture, or the mesh has no UV coordinates.
  """
  if mj_model.geom_type[geom_idx] != mjtGeom.mjGEOM_MESH:
    return -1

  matid = mj_model.geom_matid[geom_idx]
  if matid < 0 or matid >= mj_model.nmat:
    return -1

  texid = _get_texture_id(mj_model, matid)
  if texid < 0:
    return -1

  mesh_id = mj_model.geom_dataid[geom_idx]
  if mj_model.mesh_texcoordnum[mesh_id] <= 0:
    return -1

  return texid


def group_geoms_by_visual_compat(
  mj_model: mujoco.MjModel, geom_ids: list[int]
) -> list[list[int]]:
  """Partition geom IDs into groups that can be safely merged.

  Geoms sharing the same texture ID are grouped together. All
  untextured geoms form a single group.
  """
  groups: dict[int, list[int]] = {}
  for gid in geom_ids:
    tex_id = get_geom_texture_id(mj_model, gid)
    groups.setdefault(tex_id, []).append(gid)
  return list(groups.values())


# ------------------------------------------------------------------
# Merge / transform utilities
# ------------------------------------------------------------------


def merge_geoms(mj_model: mujoco.MjModel, geom_ids: list[int]) -> trimesh.Trimesh:
  """Merge multiple geoms into a single trimesh in local body space."""
  meshes = []
  for geom_id in geom_ids:
    if mj_model.geom_type[geom_id] == mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(mj_model, geom_id)
    else:
      mesh = create_primitive_mesh(mj_model, geom_id)

    transform = np.eye(4)
    transform[:3, :3] = vtf.SO3(mj_model.geom_quat[geom_id]).as_matrix()
    transform[:3, 3] = mj_model.geom_pos[geom_id]
    mesh.apply_transform(transform)
    meshes.append(mesh)

  if len(meshes) == 1:
    return meshes[0]
  return trimesh.util.concatenate(meshes)


def rotation_quat_from_vectors(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
  """Quaternion (wxyz) that rotates from_vec to to_vec."""
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.array([1.0, 0.0, 0.0, 0.0])

  if np.allclose(from_vec, -to_vec):
    perp = np.array([1.0, 0.0, 0.0])
    if abs(from_vec[0]) > 0.9:
      perp = np.array([0.0, 1.0, 0.0])
    axis = np.cross(from_vec, perp)
    axis = axis / np.linalg.norm(axis)
    return np.array([0.0, axis[0], axis[1], axis[2]])

  cross = np.cross(from_vec, to_vec)
  dot = np.dot(from_vec, to_vec)
  quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
  return quat / np.linalg.norm(quat)


def rotation_matrix_from_vectors(
  from_vec: np.ndarray, to_vec: np.ndarray
) -> np.ndarray:
  """3x3 rotation matrix that rotates from_vec to to_vec (Rodrigues)."""
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.eye(3)
  if np.allclose(from_vec, -to_vec):
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

  v = np.cross(from_vec, to_vec)
  s = np.linalg.norm(v)
  c = np.dot(from_vec, to_vec)
  vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def is_fixed_body(mj_model: mujoco.MjModel, body_id: int) -> bool:
  """Check if a body is fixed (welded to world, not attached to mocap)."""
  is_weld = mj_model.body_weldid[body_id] == 0
  root_id = mj_model.body_rootid[body_id]
  return bool(is_weld and mj_model.body_mocapid[root_id] < 0)


def get_body_name(mj_model: mujoco.MjModel, body_id: int) -> str:
  """Body name with fallback to ``body_{id}``."""
  name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
  return name if name else f"body_{body_id}"


def create_site_mesh(mj_model: mujoco.MjModel, site_id: int) -> trimesh.Trimesh:
  """Create a mesh for a single site."""
  size = mj_model.site_size[site_id]
  site_type = mj_model.site_type[site_id]
  rgba = mj_model.site_rgba[site_id].copy()

  if np.all(rgba == 0):
    rgba = np.array([0.5, 0.5, 0.5, 1.0])

  rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

  if site_type == mjtGeom.mjGEOM_SPHERE:
    mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
  elif site_type == mjtGeom.mjGEOM_BOX:
    mesh = trimesh.creation.box(extents=2.0 * size)
  elif site_type == mjtGeom.mjGEOM_CAPSULE:
    mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
  elif site_type == mjtGeom.mjGEOM_CYLINDER:
    mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
  elif site_type == mjtGeom.mjGEOM_ELLIPSOID:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.apply_scale(size)
  else:
    raise ValueError(f"Unsupported site type: {site_type}")

  _apply_flat_color(mesh, rgba_uint8)
  return mesh


def merge_sites(mj_model: mujoco.MjModel, site_ids: list[int]) -> trimesh.Trimesh:
  """Merge multiple sites into a single trimesh in local body space."""
  meshes = []
  for site_id in site_ids:
    mesh = create_site_mesh(mj_model, site_id)
    transform = np.eye(4)
    transform[:3, :3] = vtf.SO3(mj_model.site_quat[site_id]).as_matrix()
    transform[:3, 3] = mj_model.site_pos[site_id]
    mesh.apply_transform(transform)
    meshes.append(mesh)

  if len(meshes) == 1:
    return meshes[0]
  return trimesh.util.concatenate(meshes)


def get_site_name(mj_model: mujoco.MjModel, site_id: int) -> str:
  """Site name with fallback to ``site_{id}``."""
  name = mj_id2name(mj_model, mjtObj.mjOBJ_SITE, site_id)
  return name if name else f"site_{site_id}"


def merge_geoms_global(
  mj_model: mujoco.MjModel,
  mj_data: mujoco.MjData,
  geom_ids: list[int],
) -> trimesh.Trimesh:
  """Merge multiple geoms in world space using mj_data transforms."""
  meshes = []
  for geom_id in geom_ids:
    if mj_model.geom_type[geom_id] == mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(mj_model, geom_id)
    else:
      mesh = create_primitive_mesh(mj_model, geom_id)

    transform = np.eye(4)
    transform[:3, :3] = mj_data.geom_xmat[geom_id].reshape(3, 3)
    transform[:3, 3] = mj_data.geom_xpos[geom_id]
    mesh.apply_transform(transform)
    meshes.append(mesh)

  if not meshes:
    return trimesh.Trimesh()
  if len(meshes) == 1:
    return meshes[0]
  return trimesh.util.concatenate(meshes)
