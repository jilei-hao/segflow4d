"""Regression tests for mesh-warp aliasing.

Guards the bug where every propagated time point collapsed to one mesh: the
FireANTs handler warped meshes with ``MeshWrapper.update_vertices()``, which
mutates the wrapped polydata in place and returns ``self``. Because the
propagation strategies hand the *same* reference ``MeshWrapper`` objects to every
target's registration job, the resliced result aliased the single shared wrapper
and ``get_vertices()`` re-read already-warped vertices. On the in-process
(thread) dispatcher every time point ended up referencing one wrapper holding the
last write. The handler must warp into a fresh ``deepcopy`` instead.
"""

import numpy as np
import vtk

from segflow4d.common.types.mesh_wrapper import MeshWrapper


def _make_mesh(n=20) -> MeshWrapper:
    pts = vtk.vtkPoints()
    rng = np.random.default_rng(0)
    coords = rng.random((n, 3)) * 10.0
    for x, y, z in coords:
        pts.InsertNextPoint(float(x), float(y), float(z))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    return MeshWrapper(pd)


def test_update_vertices_mutates_in_place_and_returns_self():
    """Document the in-place contract the handler must defend against."""
    mesh = _make_mesh()
    original = mesh.get_vertices().copy()
    shifted = original + 5.0
    returned = mesh.update_vertices(shifted)
    assert returned is mesh, "update_vertices is expected to return self (in place)"
    np.testing.assert_allclose(mesh.get_vertices(), shifted, rtol=0, atol=1e-5)
    assert not np.allclose(original, shifted)


def test_deepcopy_warp_does_not_alias_or_mutate_shared_source():
    """The handler's pattern: warp a shared reference mesh into independent copies.

    Mimics two registration jobs reslicing the *same* reference mesh to different
    targets. Each result must be an independent object and the shared source must
    be left untouched — otherwise all time points collapse to one mesh.
    """
    ref = _make_mesh()
    ref_before = ref.get_vertices().copy()

    # Two "targets" warping the same shared ref (handler fix: deepcopy first).
    warp_a = ref.get_vertices() + np.array([1.0, 0.0, 0.0])
    out_a = ref.deepcopy().update_vertices(warp_a)

    warp_b = ref.get_vertices() + np.array([0.0, 2.0, 0.0])
    out_b = ref.deepcopy().update_vertices(warp_b)

    # Source is never mutated -> get_vertices() always reads the pristine ref.
    np.testing.assert_allclose(ref.get_vertices(), ref_before, rtol=0, atol=1e-6)

    # Results are independent objects with distinct, non-aliased geometry.
    assert out_a is not out_b
    assert out_a is not ref and out_b is not ref
    assert not np.allclose(out_a.get_vertices(), out_b.get_vertices())
    np.testing.assert_allclose(out_a.get_vertices(), ref_before + [1, 0, 0], atol=1e-5)
    np.testing.assert_allclose(out_b.get_vertices(), ref_before + [0, 2, 0], atol=1e-5)
