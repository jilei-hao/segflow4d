from logging import getLogger

from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.common.types.propagation_strategy_name import PropagationStrategyName
from segflow4d.common.types.registration_methods import REGISTRATION_METHODS
from segflow4d.common.types.tp_data import TPData
from segflow4d.propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
from segflow4d.registration.registration_manager import RegistrationManager

logger = getLogger(__name__)


class StarDirectRegistrationStrategy(AbstractPropagationStrategy):
    """Direct ref → target registration in star pattern, with no mask focusing.

    Identical in shape to :class:`StarPropagationStrategy`, but explicitly
    omits the per-timepoint masks that the SegFlow4D two-stage pipeline
    derives from the low-res mask propagation phase. Used as the high-res
    strategy of the ``direct_star`` combo, which skips the low-res phase
    entirely and serves as a baseline for comparing SegFlow4D against
    direct registration.
    """

    def propagate(self, tp_input_data: dict[int, TPData], options: PropagationOptions) -> dict[int, TPData]:
        registration_manager = RegistrationManager.get_instance()

        tp_list = list(tp_input_data.keys())
        ref_tp = tp_list[0]
        target_tps = tp_list[1:]

        logger.info(f"StarDirectRegistrationStrategy: propagating through time points {tp_list}")

        futures = {}
        for target_tp in target_tps:
            logger.info(f"Submitting direct registration: reference tp {ref_tp} to target tp {target_tp}")

            future = registration_manager.submit(
                REGISTRATION_METHODS.RUN_REGISTRATION_AND_RESLICE,
                img_fixed=tp_input_data[target_tp].image,
                img_moving=tp_input_data[ref_tp].image,
                img_to_reslice=tp_input_data[ref_tp].resliced_image,
                mesh_to_reslice=tp_input_data[ref_tp].segmentation_mesh,
                options=options,
                mask_fixed=None,
                mask_moving=None,
            )
            futures[target_tp] = future

        logger.info(f"Submitted {len(futures)} direct registration jobs to queue")

        results = {}
        for target_tp, future in futures.items():
            try:
                result = future.result()
                logger.info(f"Completed direct registration for target tp {target_tp}")
                results[target_tp] = result
            except Exception as e:
                logger.error(f"Direct registration failed for tp {target_tp}: {e}")
                raise

        for target_tp, result in results.items():
            tp_input_data[target_tp].resliced_image = result.resliced_image
            tp_input_data[target_tp].resliced_segmentation_mesh = result.resliced_segmentation_mesh
            # The pipeline reads .segmentation_mesh when copying results back to
            # per-timepoint output, so mirror the mesh there as well (matches
            # SASDPropagationStrategy's pattern).
            tp_input_data[target_tp].segmentation_mesh = result.resliced_segmentation_mesh
            tp_input_data[target_tp].warp_image = result.warp_image

        return tp_input_data

    def get_strategy_name(self) -> str:
        return PropagationStrategyName.STAR_DIRECT
