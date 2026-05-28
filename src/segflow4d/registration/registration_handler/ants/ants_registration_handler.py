"""
Classic ANTs (antspyx) CPU registration handler.

Implements the AbstractRegistrationHandler interface using the
``antspyx`` Python package (``import ants``).  All operations run on
CPU; no CUDA context is required.

Install the dependency::

    pip install antspyx
    # or: pip install segflow4d[ants]
"""

import logging
import os
import shutil
import tempfile
from time import time

import SimpleITK as sitk
import numpy as np

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.mesh_wrapper import MeshWrapper
from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.common.types.tp_data import TPData
from segflow4d.registration.registration_handler.abstract_registration_handler import AbstractRegistrationHandler
from segflow4d.registration.registration_handler.ants.ants_registration_options import AntsRegistrationOptions
from segflow4d.registration.registration_handler.greedy.cpu_mesh_warper import warp_mesh_vertices_cpu

logger = logging.getLogger(__name__)


class AntsRegistrationHandler(AbstractRegistrationHandler):
    """
    CPU-based registration handler backed by classic ANTs (antspyx).

    Performs registration via ``ants.registration`` (transform preset
    configurable via ``AntsRegistrationOptions.transform_type``) followed
    by ``ants.apply_transforms`` for segmentation reslicing and a CPU
    SimpleITK displacement-field transform for optional mesh warping.

    Images cross between SimpleITK and ANTsPy in memory via numpy with
    explicit axis re-ordering (SITK uses (z,y,x); ANTsPy uses (x,y,z)).
    """

    def __init__(self):
        super().__init__()
        logger.info("Initialized AntsRegistrationHandler")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_ants():
        """Lazy import so the rest of the package works without antspyx."""
        try:
            import ants  # noqa: PLC0415
            return ants
        except ImportError as exc:
            raise ImportError(
                "antspyx is required for the ants registration backend.  "
                "Install it with:  pip install antspyx  "
                "(or:  pip install segflow4d[ants])"
            ) from exc

    @staticmethod
    def _resolve_options(options) -> AntsRegistrationOptions:
        """
        Extract AntsRegistrationOptions from either a PropagationOptions
        object or a plain dict (the multiprocessing pickling path).
        """
        if isinstance(options, dict):
            backend_opts = options.get('registration_backend_options', {})
        else:
            backend_opts = options.registration_backend_options

        if isinstance(backend_opts, AntsRegistrationOptions):
            return backend_opts
        if isinstance(backend_opts, dict):
            return AntsRegistrationOptions(**backend_opts)
        raise ValueError(
            f"Expected AntsRegistrationOptions or dict for AntsRegistrationHandler, "
            f"got {type(backend_opts)}"
        )

    @staticmethod
    def _sitk_to_ants(sitk_img: sitk.Image, ants_module):
        """
        Convert a SimpleITK image to an ANTsPy image, preserving spacing,
        origin, and direction.

        SITK's GetArrayFromImage returns (z, y, x); ANTsPy expects
        (x, y, z), so we transpose.  SITK direction is a flat 9-tuple
        in row-major order; ANTsPy direction is a (3, 3) numpy array.
        """
        arr_zyx = sitk.GetArrayFromImage(sitk_img)
        # ants.from_numpy reads dims in (x, y, z) order
        arr_xyz = np.transpose(arr_zyx, (2, 1, 0)).copy()

        ants_img = ants_module.from_numpy(
            arr_xyz,
            origin=tuple(float(v) for v in sitk_img.GetOrigin()),
            spacing=tuple(float(v) for v in sitk_img.GetSpacing()),
            direction=np.array(sitk_img.GetDirection(), dtype=np.float64).reshape(3, 3),
        )
        return ants_img

    @staticmethod
    def _ants_to_sitk(ants_img) -> sitk.Image:
        """Inverse of :meth:`_sitk_to_ants`."""
        arr_xyz = ants_img.numpy()
        arr_zyx = np.transpose(arr_xyz, (2, 1, 0)).copy()
        sitk_img = sitk.GetImageFromArray(arr_zyx)
        sitk_img.SetSpacing(tuple(float(v) for v in ants_img.spacing))
        sitk_img.SetOrigin(tuple(float(v) for v in ants_img.origin))
        sitk_img.SetDirection(tuple(np.asarray(ants_img.direction, dtype=np.float64).flatten()))
        return sitk_img

    @staticmethod
    def _read_affine_matrix(transform_path: str) -> np.ndarray:
        """
        Read an ITK affine transform from a .mat file and return a 4x4
        homogeneous numpy matrix.
        """
        tx = sitk.ReadTransform(transform_path)
        affine = sitk.AffineTransform(tx)
        matrix = np.array(affine.GetMatrix(), dtype=np.float64).reshape(3, 3)
        translation = np.array(affine.GetTranslation(), dtype=np.float64)
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = matrix
        out[:3, 3] = translation
        return out

    @staticmethod
    def _extract_warp_path(fwd_transforms) -> str | None:
        """
        Return the path of the deformable warp file in an ANTs fwdtransforms
        list (the file matching ``*Warp.nii*`` but not ``*InverseWarp*``),
        or ``None`` if the registration produced no deformable component.
        """
        for path in fwd_transforms:
            base = os.path.basename(path)
            if 'Warp' in base and 'InverseWarp' not in base and base.endswith(('.nii', '.nii.gz')):
                return path
        return None

    @staticmethod
    def _extract_affine_path(fwd_transforms) -> str | None:
        """Return the .mat affine file path in fwdtransforms, or None."""
        for path in fwd_transforms:
            if path.endswith('.mat'):
                return path
        return None

    @staticmethod
    def _set_itk_threads(opts: AntsRegistrationOptions):
        """
        Set ITK's global thread cap for this process when ``opts.threads``
        is configured.  Process-pool workers each inherit the cap, so this
        avoids oversubscribing cores when multiple workers run in parallel.
        Returns the prior value so the caller can restore it.
        """
        if opts.threads is None:
            return None
        prior = os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(opts.threads)
        return prior

    @staticmethod
    def _restore_itk_threads(prior):
        if prior is None:
            os.environ.pop('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', None)
        else:
            os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = prior

    @staticmethod
    def _registration_kwargs(opts: AntsRegistrationOptions) -> dict:
        """
        Build the kwargs dict for ``ants.registration``.  Only the fields
        that map cleanly through the antspyx convenience API; the
        ``aff_*`` schedule fields are passed through and ignored by ANTs
        when the chosen transform_type doesn't have an affine stage.
        """
        kwargs = dict(
            type_of_transform=opts.transform_type,
            aff_metric=opts.metric if opts.metric in ('mattes', 'meansquares', 'GC') else 'mattes',
            syn_metric=opts.metric if opts.metric in ('CC', 'mattes', 'meansquares', 'demons') else 'mattes',
            aff_iterations=list(opts.aff_iterations),
            reg_iterations=list(opts.reg_iterations),
            aff_shrink_factors=list(opts.aff_shrink_factors),
            aff_smoothing_sigmas=list(opts.aff_smoothing_sigmas),
            grad_step=opts.grad_step,
            flow_sigma=opts.flow_sigma,
            total_sigma=opts.total_sigma,
            syn_sampling=opts.syn_sampling,
            verbose=opts.verbose,
        )
        if opts.random_seed is not None:
            kwargs['random_seed'] = opts.random_seed
        return kwargs

    # ------------------------------------------------------------------
    # AbstractRegistrationHandler stubs (mirroring Greedy's pattern;
    # the pipeline only invokes the *_and_reslice entry points)
    # ------------------------------------------------------------------

    def run_affine(self, img_fixed, img_moving, options: PropagationOptions):
        raise NotImplementedError(
            "AntsRegistrationHandler does not support run_affine() in isolation. "
            "Use run_registration_and_reslice() instead."
        )

    def run_deformable(self, img_fixed, img_moving, options: PropagationOptions):
        raise NotImplementedError(
            "AntsRegistrationHandler does not support run_deformable() in isolation. "
            "Use run_registration_and_reslice() instead."
        )

    def run_reslice_segmentation(self, img_to_reslice, img_reference, options: PropagationOptions):
        raise NotImplementedError(
            "AntsRegistrationHandler does not support run_reslice_segmentation() in isolation. "
            "Use run_registration_and_reslice() instead."
        )

    def run_reslice_mesh(self, mesh_to_reslice, img_reference, options: PropagationOptions):
        raise NotImplementedError(
            "AntsRegistrationHandler does not support run_reslice_mesh() in isolation. "
            "Use run_registration_and_reslice() instead."
        )

    # ------------------------------------------------------------------
    # Affine-only entry point
    # ------------------------------------------------------------------

    def run_affine_only(
        self,
        img_fixed: ImageWrapper,
        img_moving: ImageWrapper,
        options: PropagationOptions,
        mask_fixed: ImageWrapper | None = None,
        mask_moving: ImageWrapper | None = None,
    ) -> TPData:
        ants = self._import_ants()
        opts = self._resolve_options(options)
        prior_threads = self._set_itk_threads(opts)

        workdir = tempfile.mkdtemp(prefix="segflow_ants_aff_")
        try:
            fixed_ants = self._sitk_to_ants(img_fixed.get_data(), ants)
            moving_ants = self._sitk_to_ants(img_moving.get_data(), ants)
            mask_ants = (
                self._sitk_to_ants(mask_fixed.get_data(), ants)
                if mask_fixed is not None else None
            )

            logger.info("Starting ants affine-only registration ...")
            t0 = time()
            reg = ants.registration(
                fixed=fixed_ants,
                moving=moving_ants,
                type_of_transform='Affine',
                mask=mask_ants,
                aff_iterations=list(opts.aff_iterations),
                aff_shrink_factors=list(opts.aff_shrink_factors),
                aff_smoothing_sigmas=list(opts.aff_smoothing_sigmas),
                aff_metric=opts.metric if opts.metric in ('mattes', 'meansquares', 'GC') else 'mattes',
                verbose=opts.verbose,
                outprefix=os.path.join(workdir, 'aff_'),
                **({'random_seed': opts.random_seed} if opts.random_seed is not None else {}),
            )

            affine_path = self._extract_affine_path(reg['fwdtransforms'])
            if affine_path is None:
                raise RuntimeError("ants.registration did not produce an affine transform file")
            affine_matrix = self._read_affine_matrix(affine_path)
            logger.info(f"Ants affine-only registration completed in {time() - t0:.2f}s")
            return TPData(affine_matrix=affine_matrix)

        finally:
            self._restore_itk_threads(prior_threads)
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Deformable-and-reslice (with optional init affine)
    # ------------------------------------------------------------------

    def run_deformable_and_reslice(
        self,
        img_fixed: ImageWrapper,
        img_moving: ImageWrapper,
        img_to_reslice: ImageWrapper,
        mesh_to_reslice: MeshWrapper | None,
        options: PropagationOptions,
        init_affine_matrix=None,
        mask_fixed: ImageWrapper | None = None,
        mask_moving: ImageWrapper | None = None,
    ) -> TPData:
        """
        Run deformable registration (optionally initialised from a 4x4
        numpy affine matrix) and reslice the supplied segmentation /
        mesh.
        """
        ants = self._import_ants()
        opts = self._resolve_options(options)
        prior_threads = self._set_itk_threads(opts)

        workdir = tempfile.mkdtemp(prefix="segflow_ants_def_")
        init_affine_file: str | None = None
        try:
            if init_affine_matrix is not None:
                itk_affine = sitk.AffineTransform(3)
                itk_affine.SetMatrix(init_affine_matrix[:3, :3].flatten().tolist())
                itk_affine.SetTranslation(init_affine_matrix[:3, 3].tolist())
                init_affine_file = os.path.join(workdir, 'init_affine.mat')
                sitk.WriteTransform(itk_affine, init_affine_file)
                logger.debug(f"Wrote init_affine to {init_affine_file}")

            fixed_ants = self._sitk_to_ants(img_fixed.get_data(), ants)
            moving_ants = self._sitk_to_ants(img_moving.get_data(), ants)
            mask_ants = (
                self._sitk_to_ants(mask_fixed.get_data(), ants)
                if mask_fixed is not None else None
            )

            logger.info("Starting ants deformable registration (deformable_and_reslice) ...")
            t0 = time()

            # Use SyNOnly when an init affine is supplied so ANTs doesn't
            # re-run an internal affine stage; otherwise honour the configured
            # transform_type.
            transform_type = 'SyNOnly' if init_affine_matrix is not None else opts.transform_type
            reg_kwargs = self._registration_kwargs(opts)
            reg_kwargs['type_of_transform'] = transform_type
            if mask_ants is not None:
                reg_kwargs['mask'] = mask_ants
            if init_affine_file is not None:
                reg_kwargs['initial_transform'] = init_affine_file
            reg_kwargs['outprefix'] = os.path.join(workdir, 'def_')

            reg = ants.registration(
                fixed=fixed_ants,
                moving=moving_ants,
                **reg_kwargs,
            )
            logger.info(f"Ants deformable registration completed in {time() - t0:.2f}s")

            return self._reslice_and_package(
                reg=reg,
                ants=ants,
                opts=opts,
                fixed_ants=fixed_ants,
                img_fixed=img_fixed,
                img_to_reslice=img_to_reslice,
                mesh_to_reslice=mesh_to_reslice,
                read_affine=False,
            )
        finally:
            self._restore_itk_threads(prior_threads)
            shutil.rmtree(workdir, ignore_errors=True)

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
        Full registration (per ``opts.transform_type``) followed by
        segmentation / mesh reslicing.

        Returns
        -------
        TPData
            Contains ``resliced_image``, ``resliced_segmentation_mesh``
            (if a mesh was supplied), ``warp_image``, and
            ``affine_matrix`` (when the chosen transform_type produces
            an affine component).
        """
        ants = self._import_ants()
        opts = self._resolve_options(options)
        prior_threads = self._set_itk_threads(opts)

        workdir = tempfile.mkdtemp(prefix="segflow_ants_reg_")
        try:
            fixed_ants = self._sitk_to_ants(img_fixed.get_data(), ants)
            moving_ants = self._sitk_to_ants(img_moving.get_data(), ants)
            mask_ants = (
                self._sitk_to_ants(mask_fixed.get_data(), ants)
                if mask_fixed is not None else None
            )

            logger.info(f"Starting ants registration (type={opts.transform_type}) ...")
            t0 = time()

            reg_kwargs = self._registration_kwargs(opts)
            if mask_ants is not None:
                reg_kwargs['mask'] = mask_ants
            reg_kwargs['outprefix'] = os.path.join(workdir, 'reg_')

            reg = ants.registration(
                fixed=fixed_ants,
                moving=moving_ants,
                **reg_kwargs,
            )
            logger.info(f"Ants registration completed in {time() - t0:.2f}s")

            return self._reslice_and_package(
                reg=reg,
                ants=ants,
                opts=opts,
                fixed_ants=fixed_ants,
                img_fixed=img_fixed,
                img_to_reslice=img_to_reslice,
                mesh_to_reslice=mesh_to_reslice,
                read_affine=True,
            )
        finally:
            self._restore_itk_threads(prior_threads)
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Shared reslice + packaging
    # ------------------------------------------------------------------

    def _reslice_and_package(
        self,
        reg,
        ants,
        opts: AntsRegistrationOptions,
        fixed_ants,
        img_fixed: ImageWrapper,
        img_to_reslice: ImageWrapper,
        mesh_to_reslice: MeshWrapper | None,
        read_affine: bool,
    ) -> TPData:
        # ----------------------------------------------------------------
        # Reslice segmentation label map via ants.apply_transforms
        # ----------------------------------------------------------------
        logger.info("Reslicing segmentation ...")
        t0 = time()
        seg_sitk = img_to_reslice.get_data()
        reslice_pixel_id = seg_sitk.GetPixelID()
        seg_ants = self._sitk_to_ants(seg_sitk, ants)

        warped_seg_ants = ants.apply_transforms(
            fixed=fixed_ants,
            moving=seg_ants,
            transformlist=reg['fwdtransforms'],
            interpolator=opts.label_interpolation,
        )

        resliced_sitk = self._ants_to_sitk(warped_seg_ants)
        if resliced_sitk.GetPixelID() != reslice_pixel_id:
            resliced_sitk = sitk.Cast(resliced_sitk, reslice_pixel_id)
        resliced_image = ImageWrapper(resliced_sitk)
        logger.info(f"Segmentation reslicing completed in {time() - t0:.2f}s")

        # ----------------------------------------------------------------
        # Load forward warp into a SimpleITK vector image so downstream
        # callers (mesh warper, debug writers) can use it directly.
        # ----------------------------------------------------------------
        warp_path = self._extract_warp_path(reg['fwdtransforms'])
        warp_field_sitk: sitk.Image | None = None
        if warp_path is not None:
            warp_field_sitk = sitk.ReadImage(warp_path)

        # ----------------------------------------------------------------
        # Optional mesh warping (CPU SITK displacement transform)
        # ----------------------------------------------------------------
        resliced_mesh: MeshWrapper | None = None
        if mesh_to_reslice is not None:
            if warp_field_sitk is None:
                logger.warning(
                    "Mesh provided but no deformable warp produced "
                    "(transform_type=%s); mesh will not be warped.",
                    opts.transform_type,
                )
            else:
                logger.info("Warping mesh vertices on CPU ...")
                t1 = time()
                resliced_mesh = warp_mesh_vertices_cpu(
                    mesh_wrapper=mesh_to_reslice,
                    warp_field_sitk=warp_field_sitk,
                    img_fixed_sitk=img_fixed.get_data(),
                )
                logger.info(f"Mesh warping completed in {time() - t1:.2f}s")

        # ----------------------------------------------------------------
        # Extract affine matrix when requested and present
        # ----------------------------------------------------------------
        affine_matrix: np.ndarray | None = None
        if read_affine:
            aff_path = self._extract_affine_path(reg['fwdtransforms'])
            if aff_path is not None:
                affine_matrix = self._read_affine_matrix(aff_path)

        return TPData(
            resliced_image=resliced_image,
            resliced_segmentation_mesh=resliced_mesh,
            warp_image=ImageWrapper(warp_field_sitk) if warp_field_sitk is not None else None,
            affine_matrix=affine_matrix,
        )

    def get_device_type(self) -> str:
        return "cpu"
