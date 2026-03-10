from dataclasses import dataclass, field
from typing import Literal

from segflow4d.common.types.abstract_registration_options import AbstractRegistrationOptions


@dataclass
class GreedyRegistrationOptions(AbstractRegistrationOptions):
    """
    Registration options for the Greedy (picsl-greedy) CPU registration backend.

    All options map directly to greedy CLI flags. Refer to the greedy documentation
    (https://greedy.readthedocs.io/en/latest/) for detailed descriptions.

    Attributes:
        metric: Similarity metric. One of 'NCC' (normalized cross-correlation),
            'SSD' (sum of squared differences), or 'NMI' (normalized mutual information).
        metric_radius: Patch radius for NCC/WNCC metric, one value per image dimension.
            Ignored when metric is 'SSD' or 'NMI'.
        affine_dof: Degrees of freedom for affine registration.
            6 = rigid, 7 = similarity, 12 = full affine.
        affine_iterations: Number of iterations per resolution level for the affine stage.
            Each entry corresponds to one level (coarse to fine).
        deformable_iterations: Number of iterations per resolution level for the deformable stage.
        smooth_sigma_pre_mm: Pre-smoothing Gaussian sigma applied to gradient fields (mm).
            Passed to greedy as the first argument of the -s flag.
        smooth_sigma_post_mm: Post-smoothing Gaussian sigma applied to warp fields (mm).
            Passed to greedy as the second argument of the -s flag.
        jitter: Half-range of the uniform random jitter applied to voxel sample locations
            during affine registration (``-jitter`` flag).  Randomising sample positions
            avoids spurious local minima and generally improves convergence.
            Greedy's built-in default is 0.5; set to 0.0 to disable.
        threads: Number of CPU threads for greedy to use.
            None means greedy will use all available cores.
        verbosity: Greedy verbosity level (0 = silent, 1 = normal, 2 = verbose).
    """

    metric: Literal['NCC', 'SSD', 'NMI'] = 'NCC'
    metric_radius: list[int] = field(default_factory=lambda: [2, 2, 2])
    affine_dof: int = 12
    affine_iterations: list[int] = field(default_factory=lambda: [200, 100, 50])
    deformable_iterations: list[int] = field(default_factory=lambda: [200, 100, 25])
    smooth_sigma_pre_mm: float = 2.0
    smooth_sigma_post_mm: float = 0.5
    jitter: float = 0.5
    threads: int | None = None
    verbosity: int = 0

    def __post_init__(self):
        if self.metric not in ('NCC', 'SSD', 'NMI'):
            raise ValueError(f"metric must be 'NCC', 'SSD', or 'NMI', got '{self.metric}'")

        if not self.affine_iterations:
            raise ValueError("affine_iterations must be a non-empty list")

        if not self.deformable_iterations:
            raise ValueError("deformable_iterations must be a non-empty list")

        if self.affine_dof not in (6, 7, 12):
            raise ValueError(f"affine_dof must be 6, 7, or 12, got {self.affine_dof}")

        if self.smooth_sigma_pre_mm < 0:
            raise ValueError("smooth_sigma_pre_mm must be non-negative")

        if self.smooth_sigma_post_mm < 0:
            raise ValueError("smooth_sigma_post_mm must be non-negative")

        if self.jitter < 0:
            raise ValueError("jitter must be non-negative")

        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be a positive integer or None")

        if self.verbosity not in (0, 1, 2):
            raise ValueError(f"verbosity must be 0, 1, or 2, got {self.verbosity}")

        if self.metric in ('NCC',) and not self.metric_radius:
            raise ValueError("metric_radius must be provided when metric is 'NCC'")

    def affine_schedule(self) -> str:
        """Return the multi-resolution schedule string for the affine stage (e.g. '200x100x50')."""
        return 'x'.join(str(n) for n in self.affine_iterations)

    def deformable_schedule(self) -> str:
        """Return the multi-resolution schedule string for the deformable stage."""
        return 'x'.join(str(n) for n in self.deformable_iterations)

    def metric_flag(self) -> str:
        """Return the full -m flag value, including radius for NCC (e.g. 'NCC 2x2x2')."""
        if self.metric == 'NCC':
            radius_str = 'x'.join(str(r) for r in self.metric_radius)
            return f'NCC {radius_str}'
        return self.metric
