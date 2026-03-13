import numpy as np
from logging import getLogger

from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.common.types.propagation_strategy_name import PropagationStrategyName
from segflow4d.common.types.registration_methods import REGISTRATION_METHODS
from segflow4d.common.types.tp_data import TPData
from segflow4d.propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
from segflow4d.registration.registration_manager import RegistrationManager

logger = getLogger(__name__)


class SASDPropagationStrategy(AbstractPropagationStrategy):
    """
    Sequential Affine Star Deformable (SASD) propagation strategy.

    Phase 1 — Sequential affine (blocking, frame-to-frame):
        Runs affine-only registrations between adjacent timepoints in order,
        storing the resulting 4x4 affine matrix on each target TPData.

    Phase 2 — Compose affines for each target:
        For target at position k in tp_list, composes the chain
        A_k @ A_{k-1} @ ... @ A_1  (rightmost applied first)
        to produce a single affine initializer from ref → target.

    Phase 3 — Star deformable (parallel, ref → each target):
        Submits deformable registration jobs for all targets in parallel,
        each initialized with its composed affine. Collects results and
        updates TPData.
    """

    def propagate(self, tp_input_data: dict[int, TPData], options: PropagationOptions) -> dict[int, TPData]:
        registration_manager = RegistrationManager.get_instance()

        tp_list = list(tp_input_data.keys())
        ref_tp = tp_list[0]

        logger.info(f"SASDPropagationStrategy: propagating through time points {tp_list}")

        # Initialize seed tp's resliced_image if not set
        if tp_input_data[ref_tp].resliced_image is None:
            tp_input_data[ref_tp].resliced_image = tp_input_data[ref_tp].image

        # ------------------------------------------------------------------
        # Phase 1: Sequential affine
        # ------------------------------------------------------------------
        logger.info("SASD Phase 1: sequential affine registrations")
        for i in range(1, len(tp_list)):
            src_tp = tp_list[i - 1]
            tgt_tp = tp_list[i]
            logger.info(f"  Affine: {src_tp} -> {tgt_tp}")

            future = registration_manager.submit(
                REGISTRATION_METHODS.RUN_AFFINE_ONLY,
                img_fixed=tp_input_data[tgt_tp].image,
                img_moving=tp_input_data[src_tp].image,
                options=options,
                mask_fixed=tp_input_data[tgt_tp].mask,
                mask_moving=tp_input_data[src_tp].mask,
            )
            result = future.result()  # block until complete
            tp_input_data[tgt_tp].affine_matrix = result.affine_matrix
            logger.info(f"  Affine {src_tp}->{tgt_tp} done")

        # ------------------------------------------------------------------
        # Phase 2: Compose affines from ref to each target
        # ------------------------------------------------------------------
        logger.info("SASD Phase 2: composing affines")
        composed_affines: dict[int, np.ndarray] = {}
        for k in range(1, len(tp_list)):
            target_tp = tp_list[k]
            # Start with the affine for the first step (ref → tp_list[1])
            composed = tp_input_data[tp_list[1]].affine_matrix
            # Compose with each subsequent step
            for j in range(2, k + 1):
                composed = tp_input_data[tp_list[j]].affine_matrix @ composed
            composed_affines[target_tp] = composed
            logger.debug(f"  Composed affine for target {target_tp}: shape {composed.shape}")

        # ------------------------------------------------------------------
        # Phase 3: Parallel deformable (star) from ref to each target
        # ------------------------------------------------------------------
        logger.info("SASD Phase 3: parallel deformable registrations (star)")
        futures = {}
        for target_tp in tp_list[1:]:
            logger.info(f"  Submitting deformable: ref {ref_tp} -> target {target_tp}")
            future = registration_manager.submit(
                REGISTRATION_METHODS.RUN_DEFORMABLE_AND_RESLICE,
                img_fixed=tp_input_data[target_tp].image,
                img_moving=tp_input_data[ref_tp].image,
                img_to_reslice=tp_input_data[ref_tp].resliced_image,
                mesh_to_reslice=tp_input_data[ref_tp].segmentation_mesh,
                options=options,
                init_affine_matrix=composed_affines[target_tp],
                mask_fixed=tp_input_data[target_tp].mask,
                mask_moving=tp_input_data[ref_tp].mask,
            )
            futures[target_tp] = future

        logger.info(f"Submitted {len(futures)} deformable jobs")

        for target_tp, future in futures.items():
            result = future.result()
            logger.info(f"  Deformable completed for target {target_tp}")
            tp_input_data[target_tp].resliced_image = result.resliced_image
            tp_input_data[target_tp].resliced_segmentation_mesh = result.resliced_segmentation_mesh
            tp_input_data[target_tp].warp_image = result.warp_image

        logger.info("SASD propagation completed")
        return tp_input_data

    def get_strategy_name(self) -> str:
        return PropagationStrategyName.SASD
