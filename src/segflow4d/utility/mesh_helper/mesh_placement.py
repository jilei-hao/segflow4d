"""Check that a caller's mesh sits inside the image it is meant to follow.

A mesh vertex outside the image volume has no deformation to follow. Each backend
leaves such a vertex where it is, so a mesh supplied in the wrong frame is copied
unchanged to every timepoint, and nothing reports it. The common cause is a RAS
mesh: ITK-SNAP and many VTK tools write vertices in RAS, but segflow4d works in ITK
(LPS) physical coordinates, as SimpleITK reports them.
"""

from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

_LPS_RAS_FLIP = np.array([-1.0, -1.0, 1.0])


def fraction_inside(points: np.ndarray, image: sitk.Image) -> float:
    """Fraction of ``points`` (N x 3, physical LPS) inside the image volume.

    The volume runs to the outer voxel edges, i.e. continuous index -0.5 .. size-0.5.
    """
    if len(points) == 0:
        return 0.0
    origin = np.asarray(image.GetOrigin()[:3], dtype=float)
    spacing = np.asarray(image.GetSpacing()[:3], dtype=float)
    dim = image.GetDimension()
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(dim, dim)[:3, :3]
    size = np.asarray(image.GetSize()[:3], dtype=float)
    index = np.linalg.solve(direction, (np.asarray(points, dtype=float) - origin).T).T / spacing
    inside = np.all((index >= -0.5) & (index <= size - 0.5), axis=1)
    return float(inside.mean())


def check_mesh_placement(name: str, points: np.ndarray, image: sitk.Image) -> None:
    """Raise if no vertex of mesh ``name`` lies inside ``image``; warn if some do not.

    ``image`` is any image on the propagation grid (a reference segmentation or a
    frame of the 4D image).
    """
    inside = fraction_inside(points, image)
    if inside == 0.0:
        hint = ""
        if len(points) and fraction_inside(np.asarray(points) * _LPS_RAS_FLIP, image) > 0.5:
            hint = (
                " Its vertices do fall inside the image once x and y are negated, so it "
                "looks like a RAS mesh (as ITK-SNAP and many VTK tools write them). "
                "segflow4d expects ITK/LPS physical coordinates: negate the x and y "
                "coordinate of every vertex."
            )
        raise ValueError(
            f"Additional mesh {name!r} lies entirely outside the reference image, so it "
            f"cannot follow the image motion; it would be copied unchanged to every "
            f"timepoint.{hint}"
        )
    if inside < 1.0:
        logger.warning(
            f"Additional mesh {name!r}: {1.0 - inside:.1%} of its vertices lie outside the "
            f"reference image. Those vertices cannot follow the image motion."
        )
