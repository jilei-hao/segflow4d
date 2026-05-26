from dataclasses import dataclass, field
from typing import Literal

from segflow4d.common.types.abstract_registration_options import AbstractRegistrationOptions


_VALID_TRANSFORM_TYPES = (
    'Rigid',
    'Affine',
    'SyN',
    'SyNRA',
    'SyNOnly',
    'SyNAggro',
    'SyNCC',
    'SyNBold',
    'SyNBoldAff',
    'TRSAA',
    'antsRegistrationSyN[s]',
    'antsRegistrationSyN[r]',
    'antsRegistrationSyN[a]',
    'antsRegistrationSyNQuick[s]',
)

_VALID_METRICS = ('CC', 'MI', 'meansquares', 'demons', 'mattes', 'GC')

_VALID_LABEL_INTERPOLATORS = (
    'genericLabel',
    'nearestNeighbor',
    'multiLabel',
)


@dataclass
class AntsRegistrationOptions(AbstractRegistrationOptions):
    """
    Registration options for the classic ANTs (antspyx) CPU registration backend.

    These map onto the ``ants.registration`` and ``ants.apply_transforms``
    keyword arguments.  See:

        https://antspyx.readthedocs.io/en/latest/registration.html

    Attributes:
        transform_type: Preset passed as ``type_of_transform`` to
            ``ants.registration``.  ``'SyN'`` (default) is the classic
            symmetric normalisation deformable preset; ``'SyNRA'`` adds
            an initial rigid+affine stage for unaligned inputs;
            ``'antsRegistrationSyN[s]'`` uses the Tustison canonical
            preset (slowest, highest quality).
        metric: Similarity metric for the deformable stage.
            One of ``'CC'``, ``'MI'``, ``'meansquares'``, ``'demons'``,
            ``'mattes'``, ``'GC'``.  ANTs' default is ``'mattes'``.
        aff_iterations: Iteration schedule for the affine stage,
            coarse to fine (e.g. ``(2100, 1200, 1200, 100)``).
        reg_iterations: Iteration schedule for the deformable (SyN) stage,
            coarse to fine (e.g. ``(40, 20, 0)``).
        aff_shrink_factors: Multi-resolution shrink factors for the
            affine stage.
        aff_smoothing_sigmas: Gaussian smoothing sigmas (voxels) for
            the affine stage.
        grad_step: Gradient descent step size for the SyN optimiser
            (``ants.registration`` ``grad_step``).
        flow_sigma: Sigma for the regularising Gaussian on the update
            field (voxels).
        total_sigma: Sigma for the regularising Gaussian on the total
            (accumulated) field (voxels).  ANTs default is ``0.0``.
        syn_sampling: Sampling parameter for the deformable metric
            (e.g. radius for CC, bins for MI).
        label_interpolation: Interpolator used by ``ants.apply_transforms``
            for label maps.  ``'genericLabel'`` is the recommended
            default for multi-label segmentations.
        threads: Number of CPU threads ANTs/ITK should use.  When set,
            the handler exports ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS``
            for the duration of the call so concurrent process workers
            do not oversubscribe cores.  ``None`` lets ITK pick.
        random_seed: Optional seed for ANTs random sampling
            (``ants.registration`` ``random_seed``).
        verbose: When True, pass ``verbose=True`` to ``ants.registration``.
    """

    transform_type: Literal[
        'Rigid', 'Affine', 'SyN', 'SyNRA', 'SyNOnly', 'SyNAggro',
        'SyNCC', 'SyNBold', 'SyNBoldAff', 'TRSAA',
        'antsRegistrationSyN[s]', 'antsRegistrationSyN[r]',
        'antsRegistrationSyN[a]', 'antsRegistrationSyNQuick[s]',
    ] = 'SyN'
    metric: Literal['CC', 'MI', 'meansquares', 'demons', 'mattes', 'GC'] = 'mattes'
    aff_iterations: tuple[int, ...] = (2100, 1200, 1200, 100)
    reg_iterations: tuple[int, ...] = (40, 20, 0)
    aff_shrink_factors: tuple[int, ...] = (6, 4, 2, 1)
    aff_smoothing_sigmas: tuple[int, ...] = (3, 2, 1, 0)
    grad_step: float = 0.1
    flow_sigma: float = 3.0
    total_sigma: float = 0.0
    syn_sampling: int = 32
    label_interpolation: Literal['genericLabel', 'nearestNeighbor', 'multiLabel'] = 'genericLabel'
    threads: int | None = None
    random_seed: int | None = None
    verbose: bool = False

    def __post_init__(self):
        if self.transform_type not in _VALID_TRANSFORM_TYPES:
            raise ValueError(
                f"transform_type must be one of {_VALID_TRANSFORM_TYPES}, "
                f"got '{self.transform_type}'"
            )

        if self.metric not in _VALID_METRICS:
            raise ValueError(
                f"metric must be one of {_VALID_METRICS}, got '{self.metric}'"
            )

        if self.label_interpolation not in _VALID_LABEL_INTERPOLATORS:
            raise ValueError(
                f"label_interpolation must be one of {_VALID_LABEL_INTERPOLATORS}, "
                f"got '{self.label_interpolation}'"
            )

        # dict-loaded options often come in as lists; coerce to tuples
        # so the dataclass stays comparable and immutable-ish.
        if not isinstance(self.aff_iterations, tuple):
            self.aff_iterations = tuple(self.aff_iterations)
        if not isinstance(self.reg_iterations, tuple):
            self.reg_iterations = tuple(self.reg_iterations)
        if not isinstance(self.aff_shrink_factors, tuple):
            self.aff_shrink_factors = tuple(self.aff_shrink_factors)
        if not isinstance(self.aff_smoothing_sigmas, tuple):
            self.aff_smoothing_sigmas = tuple(self.aff_smoothing_sigmas)

        if not self.aff_iterations:
            raise ValueError("aff_iterations must be a non-empty sequence")

        if not self.reg_iterations:
            raise ValueError("reg_iterations must be a non-empty sequence")

        if any(n < 0 for n in self.aff_iterations):
            raise ValueError("aff_iterations entries must be non-negative")

        if any(n < 0 for n in self.reg_iterations):
            raise ValueError("reg_iterations entries must be non-negative")

        if self.grad_step <= 0:
            raise ValueError("grad_step must be positive")

        if self.flow_sigma < 0:
            raise ValueError("flow_sigma must be non-negative")

        if self.total_sigma < 0:
            raise ValueError("total_sigma must be non-negative")

        if self.syn_sampling <= 0:
            raise ValueError("syn_sampling must be positive")

        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be a positive integer or None")
