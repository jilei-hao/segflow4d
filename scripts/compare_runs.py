#!/usr/bin/env python
"""Compare two segflow4d output directories: a frozen reference run and a candidate.

This is the refactor gate. A behaviour-preserving change must reproduce the
frozen run's segmentations and images exactly, and its meshes to within the
run-to-run noise floor. When the outputs differ, the per-timepoint Dice and
volume ratios show how far they moved.

Why meshes get a tolerance: on Ludwig CUDA the label maps are bit-identical
from run to run, but warped mesh vertices are not. They differ by about 1e-4 mm
(float32 GPU reductions add up in a different order each run). Label
interpolation absorbs that difference; continuous vertex positions don't.
Set --mesh-tol-mm from a measured same-code rerun, not by guess.

Voxel data and mesh point coordinates are compared, not file bytes: gzip headers
carry timestamps, so the files differ even when the data is identical.

Usage:
    compare_runs.py REFERENCE_DIR CANDIDATE_DIR [--mesh-tol-mm TOL] [--json OUT.json] [--quiet]

Verdicts:
    IDENTICAL   every compared output is exactly equal
    EQUIVALENT  seg-4d and image-4d are exactly equal, and every mesh has the same
                topology with no vertex further than TOL mm from its reference
    DIFFERENT   anything else

Exit status:
    0  IDENTICAL or EQUIVALENT (the gate passes)
    1  DIFFERENT
    2  the runs are not comparable (missing outputs, different shape or geometry)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import SimpleITK as sitk

SEG_4D = "seg-4d.nii.gz"
IMAGE_4D = "image-4d.nii.gz"
MESH_DIRS = ("segmentation-mesh", "additional-mesh")
TP_RE = re.compile(r"_tp-(\d+)\.vtp$")


class NotComparable(Exception):
    pass


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def _read_image(path: str) -> sitk.Image:
    if not os.path.isfile(path):
        raise NotComparable(f"missing: {path}")
    return sitk.ReadImage(path)


def _same_geometry(a: sitk.Image, b: sitk.Image) -> list[str]:
    problems = []
    if a.GetSize() != b.GetSize():
        problems.append(f"size {a.GetSize()} vs {b.GetSize()}")
    for name in ("GetSpacing", "GetOrigin", "GetDirection"):
        va, vb = np.asarray(getattr(a, name)()), np.asarray(getattr(b, name)())
        if not np.allclose(va, vb, rtol=0, atol=1e-6):
            problems.append(f"{name[3:].lower()} {tuple(va)} vs {tuple(vb)}")
    return problems


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2.0 * int(np.logical_and(a, b).sum()) / total if total else float("nan")


def _frame_tps(run_dir: str, n_frames: int) -> list[str]:
    """Label the frames of seg-4d by timepoint where that can be recovered.

    seg-4d holds only the processed timepoints, in ascending order, and carries no
    timepoint index (KNOWN_ISSUES R3), so the timepoints are read back from the
    segmentation-mesh filenames. Falls back to frame indices if that fails."""
    mesh_dir = os.path.join(run_dir, "segmentation-mesh")
    if os.path.isdir(mesh_dir):
        tps = sorted(int(m.group(1)) for f in os.listdir(mesh_dir) if (m := TP_RE.search(f)))
        if len(tps) == n_frames:
            return [f"tp{tp:03d}" for tp in tps]
    return [f"frame{i:03d}" for i in range(n_frames)]


def compare_seg4d(ref_dir: str, cand_dir: str) -> dict:
    ref_img = _read_image(os.path.join(ref_dir, SEG_4D))
    cand_img = _read_image(os.path.join(cand_dir, SEG_4D))
    problems = _same_geometry(ref_img, cand_img)
    if problems:
        raise NotComparable(f"{SEG_4D}: " + "; ".join(problems))

    ref = sitk.GetArrayFromImage(ref_img)    # (t, z, y, x)
    cand = sitk.GetArrayFromImage(cand_img)
    labels = sorted(int(v) for v in np.union1d(np.unique(ref), np.unique(cand)) if v != 0)

    frames = []
    for name, r, c in zip(_frame_tps(ref_dir, ref.shape[0]), ref, cand):
        ref_fg, cand_fg = r > 0, c > 0
        frames.append({
            "frame": name,
            "identical": bool(np.array_equal(r, c)),
            "voxels_changed": int(np.count_nonzero(r != c)),
            "dice_fg": _dice(ref_fg, cand_fg),
            "dice_by_label": {str(l): _dice(r == l, c == l) for l in labels},
            "volume_ratio_fg": (float(cand_fg.sum()) / ref_fg.sum()) if ref_fg.any() else float("nan"),
        })
    return {"identical": all(f["identical"] for f in frames), "labels": labels, "frames": frames}


def compare_image4d(ref_dir: str, cand_dir: str) -> dict:
    ref_path, cand_path = os.path.join(ref_dir, IMAGE_4D), os.path.join(cand_dir, IMAGE_4D)
    if not (os.path.isfile(ref_path) and os.path.isfile(cand_path)):
        return {"identical": None, "note": "image-4d missing in one run; not compared"}
    ref_img, cand_img = sitk.ReadImage(ref_path), sitk.ReadImage(cand_path)
    problems = _same_geometry(ref_img, cand_img)
    if problems:
        return {"identical": False, "note": "; ".join(problems)}
    same = np.array_equal(sitk.GetArrayViewFromImage(ref_img), sitk.GetArrayViewFromImage(cand_img))
    return {"identical": bool(same)}


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------

def _mesh_files(run_dir: str) -> dict[str, str]:
    """Relative path -> absolute path for every .vtp under the mesh output dirs."""
    found = {}
    for sub in MESH_DIRS:
        root = os.path.join(run_dir, sub)
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".vtp"):
                    full = os.path.join(dirpath, f)
                    found[os.path.relpath(full, run_dir)] = full
    return found


def _read_points(path: str) -> tuple[np.ndarray, int]:
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    poly = reader.GetOutput()
    points = poly.GetPoints()
    coords = vtk_to_numpy(points.GetData()) if points is not None else np.empty((0, 3))
    return np.asarray(coords, dtype=np.float64), poly.GetNumberOfCells()


def compare_meshes(ref_dir: str, cand_dir: str, tol_mm: float = 0.0) -> dict:
    ref_files, cand_files = _mesh_files(ref_dir), _mesh_files(cand_dir)
    only_ref = sorted(set(ref_files) - set(cand_files))
    only_cand = sorted(set(cand_files) - set(ref_files))
    meshes = []
    for rel in sorted(set(ref_files) & set(cand_files)):
        ref_pts, ref_cells = _read_points(ref_files[rel])
        cand_pts, cand_cells = _read_points(cand_files[rel])
        same_topology = ref_pts.shape == cand_pts.shape and ref_cells == cand_cells
        max_dist = (float(np.max(np.linalg.norm(ref_pts - cand_pts, axis=1)))
                    if same_topology and len(ref_pts) else (0.0 if same_topology else None))
        meshes.append({
            "mesh": rel,
            "identical": bool(same_topology and np.array_equal(ref_pts, cand_pts)),
            "within_tolerance": bool(same_topology and max_dist is not None and max_dist <= tol_mm),
            "same_topology": same_topology,
            "max_point_distance": max_dist,
        })
    same_set = not only_ref and not only_cand
    distances = [m["max_point_distance"] for m in meshes if m["max_point_distance"] is not None]
    return {
        "identical": same_set and all(m["identical"] for m in meshes),
        "within_tolerance": same_set and all(m["within_tolerance"] for m in meshes),
        "tolerance_mm": tol_mm,
        "max_point_distance": max(distances) if distances else None,
        "only_in_reference": only_ref,
        "only_in_candidate": only_cand,
        "meshes": meshes,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def compare(ref_dir: str, cand_dir: str, mesh_tol_mm: float = 0.0) -> dict:
    seg = compare_seg4d(ref_dir, cand_dir)
    image = compare_image4d(ref_dir, cand_dir)
    meshes = compare_meshes(ref_dir, cand_dir, mesh_tol_mm)
    exact_volumes = bool(seg["identical"] and image["identical"] is not False)
    if exact_volumes and meshes["identical"]:
        verdict = "IDENTICAL"
    elif exact_volumes and meshes["within_tolerance"]:
        verdict = "EQUIVALENT"
    else:
        verdict = "DIFFERENT"
    return {
        "reference": os.path.abspath(ref_dir),
        "candidate": os.path.abspath(cand_dir),
        "verdict": verdict,
        "identical": verdict == "IDENTICAL",
        "seg_4d": seg,
        "image_4d": image,
        "meshes": meshes,
    }


def _print_report(report: dict) -> None:
    seg, meshes = report["seg_4d"], report["meshes"]
    print(f"reference: {report['reference']}")
    print(f"candidate: {report['candidate']}")
    print(f"{'frame':>9} {'same':>5} {'changed':>9} {'Dice(fg)':>9} {'vol ratio':>10}")
    for f in seg["frames"]:
        print(f"{f['frame']:>9} {'yes' if f['identical'] else 'NO':>5} {f['voxels_changed']:>9} "
              f"{f['dice_fg']:>9.4f} {f['volume_ratio_fg']:>10.4f}")
    differing = [m for m in meshes["meshes"] if not m["identical"]]
    print(f"meshes: {len(meshes['meshes'])} compared, {len(differing)} not bit-identical "
          f"(largest vertex distance {meshes['max_point_distance']} mm, tolerance "
          f"{meshes['tolerance_mm']} mm), "
          f"{len(meshes['only_in_reference'])} only in reference, "
          f"{len(meshes['only_in_candidate'])} only in candidate")
    for m in differing[:10]:
        print(f"  {m['mesh']}: topology {'same' if m['same_topology'] else 'DIFFERENT'}, "
              f"max point distance {m['max_point_distance']}")
    print(f"image-4d identical: {report['image_4d']['identical']}")
    print("RESULT:", report["verdict"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--mesh-tol-mm", type=float, default=0.0,
                        help="largest vertex distance (mm) still counted as equivalent; "
                             "set it from a measured same-code rerun (default 0: exact)")
    parser.add_argument("--json", help="also write the full report here")
    parser.add_argument("--quiet", action="store_true", help="print only the result line")
    args = parser.parse_args(argv)

    try:
        report = compare(args.reference_dir, args.candidate_dir, args.mesh_tol_mm)
    except NotComparable as exc:
        print(f"NOT COMPARABLE: {exc}", file=sys.stderr)
        return 2

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    if args.quiet:
        print("RESULT:", report["verdict"])
    else:
        _print_report(report)
    return 0 if report["verdict"] in ("IDENTICAL", "EQUIVALENT") else 1


if __name__ == "__main__":
    sys.exit(main())
