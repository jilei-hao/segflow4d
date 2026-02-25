"""
Mesh-to-mesh MSD (mean symmetric surface distance) and HD95 for vtkPolyData,
using area-weighted surface sampling and closest-point distance to the other surface.

Requirements:
  pip install vtk numpy

Notes:
- Works with vtkPolyData (triangulates internally).
- Uses vtkImplicitPolyDataDistance (correspondence-free).
- Sampling is area-weighted by triangle area (stable across remeshing/decimation).
"""

from __future__ import annotations
import numpy as np
import vtk


# ----------------------------
# Helpers: VTK <-> NumPy
# ----------------------------
def _triangulate(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(poly)
    tf.Update()
    out = vtk.vtkPolyData()
    out.ShallowCopy(tf.GetOutput())
    return out


def surface_area_mm2(poly: vtk.vtkPolyData) -> float:
    tri = _triangulate(poly)
    mp = vtk.vtkMassProperties()
    mp.SetInputData(tri)
    mp.Update()
    return float(mp.GetSurfaceArea())


def _sample_points_area_weighted(poly: vtk.vtkPolyData, n_points: int, seed: int = 0) -> np.ndarray:
    """
    Area-weighted sampling on a triangulated vtkPolyData surface.
    Returns (n_points, 3) points in the same physical units as the mesh (typically mm).
    """
    rng = np.random.default_rng(seed)
    tri = _triangulate(poly)

    pts = tri.GetPoints()
    polys = tri.GetPolys()

    # Extract triangles into an (n_tri, 3) array of point IDs
    polys.InitTraversal()
    id_list = vtk.vtkIdList()

    tri_ids = []
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() != 3:
            continue
        tri_ids.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])

    if not tri_ids:
        raise ValueError("No triangles found in vtkPolyData after triangulation.")

    tri_ids = np.asarray(tri_ids, dtype=np.int64)
    n_tri = tri_ids.shape[0]

    # Get triangle vertices
    # (Looping over vtk points is OK here; if you have huge meshes, we can optimize further.)
    v0 = np.empty((n_tri, 3), dtype=np.float64)
    v1 = np.empty((n_tri, 3), dtype=np.float64)
    v2 = np.empty((n_tri, 3), dtype=np.float64)
    for i in range(n_tri):
        v0[i] = pts.GetPoint(int(tri_ids[i, 0]))
        v1[i] = pts.GetPoint(int(tri_ids[i, 1]))
        v2[i] = pts.GetPoint(int(tri_ids[i, 2]))

    # Triangle areas
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    area_sum = areas.sum()
    if area_sum <= 0:
        raise ValueError("Surface area is zero; cannot sample points.")

    probs = areas / area_sum

    # Choose triangles proportional to area
    tri_idx = rng.choice(n_tri, size=n_points, replace=True, p=probs)

    # Sample uniformly within each chosen triangle using barycentric coordinates
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    sqrt_r1 = np.sqrt(r1)

    a = v0[tri_idx]
    b = v1[tri_idx]
    c = v2[tri_idx]

    samples = (1 - sqrt_r1)[:, None] * a + (sqrt_r1 * (1 - r2))[:, None] * b + (sqrt_r1 * r2)[:, None] * c
    return samples


def choose_n_by_target_spacing(poly: vtk.vtkPolyData, target_spacing_mm: float,
                               n_min: int = 20000, n_max: int = 150000) -> int:
    """
    Choose N so average sample spacing ~ target_spacing_mm:
        s ≈ sqrt(A / N)  =>  N ≈ A / s^2
    Then clamp to [n_min, n_max].
    """
    A = surface_area_mm2(poly)  # mm^2
    n = int(np.ceil(A / (target_spacing_mm ** 2)))
    return int(np.clip(n, n_min, n_max))


# ----------------------------
# Distances: points -> surface
# ----------------------------
def _point_to_surface_distances(points_xyz: np.ndarray, surface: vtk.vtkPolyData) -> np.ndarray:
    """
    Compute unsigned closest distance from each point to a vtkPolyData surface.
    Uses vtkImplicitPolyDataDistance (internally builds locators).
    """
    surface_tri = _triangulate(surface)
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(surface_tri)

    d = np.empty(points_xyz.shape[0], dtype=np.float64)
    for i, p in enumerate(points_xyz):
        # EvaluateFunction returns signed distance; magnitude is what we want for surface distance
        d[i] = abs(ipd.EvaluateFunction(float(p[0]), float(p[1]), float(p[2])))
    return d


def msd_and_hd95_from_polydata(
    mesh_a: vtk.vtkPolyData,
    mesh_b: vtk.vtkPolyData,
    *,
    n_points: int | None = None,
    target_spacing_mm: float = 0.5,
    seed: int = 0,
) -> tuple[float, float, dict]:
    """
    Compute:
      - MSD (mean symmetric surface distance) in mesh units (mm)
      - HD95 (95th percentile symmetric Hausdorff) in mesh units (mm)

    Returns: (msd, hd95, info_dict)
    """
    if n_points is None:
        n_a = choose_n_by_target_spacing(mesh_a, target_spacing_mm=target_spacing_mm)
        n_b = choose_n_by_target_spacing(mesh_b, target_spacing_mm=target_spacing_mm)
    else:
        n_a = n_b = int(n_points)

    # Sample surface points
    pts_a = _sample_points_area_weighted(mesh_a, n_a, seed=seed)
    pts_b = _sample_points_area_weighted(mesh_b, n_b, seed=seed + 1)

    # Closest distances (bidirectional)
    d_a_to_b = _point_to_surface_distances(pts_a, mesh_b)
    d_b_to_a = _point_to_surface_distances(pts_b, mesh_a)

    d_all = np.concatenate([d_a_to_b, d_b_to_a])

    msd = float(d_all.mean())
    hd95 = float(np.percentile(d_all, 95))

    info = {
        "n_points_a": n_a,
        "n_points_b": n_b,
        "target_spacing_mm": float(target_spacing_mm),
        "area_a_mm2": surface_area_mm2(mesh_a),
        "area_b_mm2": surface_area_mm2(mesh_b),
        "seed": seed,
    }
    return msd, hd95, info


# ----------------------------
# Example usage
# ----------------------------
def read_vtp(path: str) -> vtk.vtkPolyData:
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    out = vtk.vtkPolyData()
    out.ShallowCopy(r.GetOutput())
    return out


if __name__ == "__main__":
    # Example: load two surfaces (same anatomy, different segmentation)
    mesh1 = read_vtp("seg1_surface.vtp")
    mesh2 = read_vtp("seg2_surface.vtp")

    msd, hd95, info = msd_and_hd95_from_polydata(
        mesh1, mesh2,
        n_points=None,              # or set fixed n_points=100000
        target_spacing_mm=0.5,       # typical for ~1mm imaging
        seed=42
    )

    print("MSD (mm):", msd)
    print("HD95 (mm):", hd95)
    print("Info:", info)
