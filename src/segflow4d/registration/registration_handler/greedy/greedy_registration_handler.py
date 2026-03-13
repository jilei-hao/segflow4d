"""
Greedy CPU registration handler.

Implements the AbstractRegistrationHandler interface using the picsl-greedy
Python bindings (``picsl_greedy.Greedy3D``).  All operations run on CPU;
no CUDA context is required.

Install the dependency::

    pip install picsl-greedy
    # or: pip install segflow4d[greedy]
"""

import logging
from time import time

import SimpleITK as sitk
import numpy as np

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.mesh_wrapper import MeshWrapper
from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.common.types.tp_data import TPData
from segflow4d.registration.registration_handler.abstract_registration_handler import AbstractRegistrationHandler
from segflow4d.registration.registration_handler.greedy.greedy_registration_options import GreedyRegistrationOptions
from segflow4d.registration.registration_handler.greedy.cpu_mesh_warper import warp_mesh_vertices_cpu

logger = logging.getLogger(__name__)


class GreedyRegistrationHandler(AbstractRegistrationHandler):
    """
    CPU-based registration handler backed by picsl-greedy (``Greedy3D``).

    Performs affine and deformable registration followed by segmentation
    (and optionally mesh) reslicing.  All input/output images are exchanged
    with greedy as in-memory SimpleITK objects — no temporary files are written.
    """

    def __init__(self):
        super().__init__()
        logger.info("Initialized GreedyRegistrationHandler")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_greedy():
        """Lazy import so the rest of the package works without picsl-greedy."""
        try:
            from picsl_greedy import Greedy3D  # noqa: PLC0415
            return Greedy3D
        except ImportError as exc:
            raise ImportError(
                "picsl-greedy is required for the greedy registration backend.  "
                "Install it with:  pip install picsl-greedy"
            ) from exc

    @staticmethod
    def _resolve_options(options) -> GreedyRegistrationOptions:
        """
        Extract GreedyRegistrationOptions from either a PropagationOptions object
        or a plain dict (which occurs when arguments cross process boundaries via
        multiprocessing pickling).
        """
        if isinstance(options, dict):
            backend_opts = options.get('registration_backend_options', {})
            if isinstance(backend_opts, dict):
                return GreedyRegistrationOptions(**backend_opts)
            return backend_opts
        # PropagationOptions object
        backend_opts = options.registration_backend_options
        if isinstance(backend_opts, dict):
            return GreedyRegistrationOptions(**backend_opts)
        if isinstance(backend_opts, GreedyRegistrationOptions):
            return backend_opts
        raise ValueError(
            f"Expected GreedyRegistrationOptions or dict for GreedyRegistrationHandler, "
            f"got {type(backend_opts)}"
        )

    @staticmethod
    def _build_common_flags(opts: GreedyRegistrationOptions) -> str:
        """Build CLI fragment shared across greedy calls (threads, verbosity, precision)."""
        parts = []
        if opts.use_float:
            parts.append("-float")
        if opts.threads is not None:
            parts.append(f"-threads {opts.threads}")
        if opts.verbosity > 0:
            parts.append(f"-V {opts.verbosity}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # AbstractRegistrationHandler stubs (individual stages not needed
    # for the two-stage pipeline; expose them as NotImplementedError so
    # subclasses can override if needed).
    # ------------------------------------------------------------------

    def run_affine(self, img_fixed, img_moving, options: PropagationOptions):
        raise NotImplementedError(
            "GreedyRegistrationHandler does not support run_affine() in isolation. "
            "Use run_registration_and_reslice() instead."
        )

    def run_deformable(self, img_fixed, img_moving, options: PropagationOptions):
        raise NotImplementedError(
            "GreedyRegistrationHandler does not support run_deformable() in isolation. "
            "Use run_registration_and_reslice() instead."
        )

    def run_reslice_segmentation(self, img_to_reslice, img_reference, options: PropagationOptions):
        raise NotImplementedError(
            "GreedyRegistrationHandler does not support run_reslice_segmentation() in isolation. "
            "Use run_registration_and_reslice() instead."
        )

    def run_reslice_mesh(self, mesh_to_reslice, img_reference, options: PropagationOptions):
        raise NotImplementedError(
            "GreedyRegistrationHandler does not support run_reslice_mesh() in isolation. "
            "Use run_registration_and_reslice() instead."
        )

    # ------------------------------------------------------------------
    # Main registration entry point
    # ------------------------------------------------------------------

    def run_registration_and_reslice(
        self,
        img_fixed: ImageWrapper,
        img_moving: ImageWrapper,
        img_to_reslice: ImageWrapper,
        mesh_to_reslice: MeshWrapper | None,
        options: PropagationOptions,
        mask_fixed: ImageWrapper | None = None,
        mask_moving: ImageWrapper | None = None,
    ) -> TPData:
        """
        Affine + deformable registration followed by segmentation / mesh reslicing.

        Pipeline
        --------
        1. Affine registration  (``-a -dof {dof}``)
        2. Deformable registration  (greedy SyN, initialized with affine)
        3. Reslice segmentation label map  (``-ri LABEL``)
        4. Warp mesh vertices  (optional, via :func:`warp_mesh_vertices_cpu`)

        Parameters
        ----------
        img_fixed:
            Fixed (target) image.
        img_moving:
            Moving (source) image.
        img_to_reslice:
            Label map to propagate (segmentation or mask).
        mesh_to_reslice:
            Surface mesh to warp; ``None`` skips mesh warping.
        options:
            Propagation options carrying a ``GreedyRegistrationOptions`` instance
            in ``registration_backend_options``.  May also be a plain ``dict``
            (multiprocessing serialisation).
        mask_fixed:
            Optional binary mask for the fixed image, passed to greedy via ``-gm``.
        mask_moving:
            Accepted for interface compatibility but intentionally ignored.
            Greedy native mask logic only applies the fixed mask.

        Returns
        -------
        TPData
            Contains ``resliced_image``, ``resliced_segmentation_mesh`` (if mesh
            provided), ``warp_image``, and ``affine_matrix``.
        """
        Greedy3D = self._import_greedy()
        opts = self._resolve_options(options)

        itk_fixed = img_fixed.get_data()
        assert itk_fixed is not None, "img_fixed contains no image data"
        itk_moving = img_moving.get_data()
        itk_to_reslice = img_to_reslice.get_data()
        reslice_pixel_id: int = img_to_reslice.get_data().GetPixelID()  # type: ignore[union-attr]

        common_flags = self._build_common_flags(opts)
        metric_str = opts.metric_flag()

        # Greedy native mask: only the fixed mask is applied via -gm; the moving
        # mask is intentionally ignored (greedy handles it internally).
        itk_mask_fixed = mask_fixed.get_data() if mask_fixed is not None else None

        # Reuse a single Greedy3D instance across all stages so that cached
        # ITK transform objects (e.g. my_affine) retain the correct ITK type
        # when referenced by subsequent commands via -it / -r flags.
        g = Greedy3D()

        # ----------------------------------------------------------------
        # Stage 1: Affine registration
        # ----------------------------------------------------------------
        logger.info("Starting greedy affine registration ...")
        t0 = time()

        affine_cmd = (
            f"-i my_fixed my_moving "
            f"-a -dof {opts.affine_dof} "
            f"-n {opts.affine_schedule()} "
            f"-m {metric_str} "
            f"-jitter {opts.jitter} "
            f"{'-gm my_mask_fixed ' if itk_mask_fixed is not None else ''}"
            f"-o my_affine "
            f"{common_flags}"
        ).strip()

        logger.debug(f"Greedy affine command: {affine_cmd}")
        affine_kwargs = dict(my_fixed=itk_fixed, my_moving=itk_moving, my_affine=None)
        if itk_mask_fixed is not None:
            affine_kwargs['my_mask_fixed'] = itk_mask_fixed
        g.execute(affine_cmd, **affine_kwargs)

        affine_matrix: np.ndarray = g['my_affine']
        logger.info(f"Affine registration completed in {time() - t0:.2f}s")

        # ----------------------------------------------------------------
        # Stage 2: Deformable registration
        # ----------------------------------------------------------------
        logger.info("Starting greedy deformable registration ...")
        t1 = time()

        deform_cmd = (
            f"-i my_fixed my_moving "
            f"-it my_affine "
            f"-n {opts.deformable_schedule()} "
            f"-m {metric_str} "
            f"-s {opts.smooth_sigma_pre_mm}mm {opts.smooth_sigma_post_mm}mm "
            f"{'-gm my_mask_fixed ' if itk_mask_fixed is not None else ''}"
            f"-o my_warp "
            f"{common_flags}"
        ).strip()

        logger.debug(f"Greedy deformable command: {deform_cmd}")
        deform_kwargs = dict(my_fixed=itk_fixed, my_moving=itk_moving, my_warp=None)
        if itk_mask_fixed is not None:
            deform_kwargs['my_mask_fixed'] = itk_mask_fixed
        g.execute(deform_cmd, **deform_kwargs)

        warp_field_sitk: sitk.Image = g['my_warp']
        logger.info(f"Deformable registration completed in {time() - t1:.2f}s")

        # ----------------------------------------------------------------
        # Stage 3: Reslice segmentation label map
        # ----------------------------------------------------------------
        logger.info("Reslicing segmentation ...")
        t2 = time()

        reslice_cmd = (
            f"-rf my_fixed "
            f"-ri LABEL {opts.label_interpolation} "
            f"-rm my_seg my_resliced "
            f"-r my_warp my_affine "
            f"{common_flags}"
        ).strip()

        logger.debug(f"Greedy reslice command: {reslice_cmd}")
        g.execute(reslice_cmd, my_fixed=itk_fixed, my_seg=itk_to_reslice, my_resliced=None)

        resliced_sitk: sitk.Image = g['my_resliced']
        # Greedy LABEL reslice may return a different pixel type (e.g. float64).
        # Cast back to the original segmentation pixel type so downstream
        # JoinSeriesImageFilter calls don't fail on type mismatches.
        if resliced_sitk.GetPixelID() != reslice_pixel_id:
            resliced_sitk = sitk.Cast(resliced_sitk, reslice_pixel_id)
        resliced_image = ImageWrapper(resliced_sitk)
        logger.info(f"Segmentation reslicing completed in {time() - t2:.2f}s")

        # ----------------------------------------------------------------
        # Stage 4: Warp mesh (optional)
        # ----------------------------------------------------------------
        resliced_mesh: MeshWrapper | None = None
        if mesh_to_reslice is not None:
            logger.info("Warping mesh vertices on CPU ...")
            t3 = time()
            resliced_mesh = warp_mesh_vertices_cpu(
                mesh_wrapper=mesh_to_reslice,
                warp_field_sitk=warp_field_sitk,
                img_fixed_sitk=itk_fixed,
            )
            logger.info(f"Mesh warping completed in {time() - t3:.2f}s")

        return TPData(
            resliced_image=resliced_image,
            resliced_segmentation_mesh=resliced_mesh,
            warp_image=ImageWrapper(warp_field_sitk),
            affine_matrix=affine_matrix,
        )

    def get_device_type(self) -> str:
        return "cpu"
