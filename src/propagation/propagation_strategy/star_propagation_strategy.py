from propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
from common.types.propagation_strategy_data import PropagationStrategyData
from registration.registration_manager import RegistrationManager
from common.types.registration_methods import REGISTRATION_METHODS
from common.types.propagation_strategy_name import PropagationStrategyName
from logging import getLogger

logger = getLogger(__name__)

class StarPropagationStrategy(AbstractPropagationStrategy):
    def propagate(self, tp_input_data: dict[int, PropagationStrategyData], options) -> dict[int, PropagationStrategyData]:
        # get all time points
        time_points = list(tp_input_data.keys())
        logger.info(f"StarPropagationStrategy: propagating through time points {time_points}")

        registration_manager = RegistrationManager.get_instance()

        # propagate from reference time point to all other time points
        reference_tp = time_points[0]
        data_ref = tp_input_data[reference_tp]
        
        # Ensure reference timepoint has resliced_image
        if data_ref.resliced_image is None:
            logger.warning(f"No resliced_image for reference timepoint {reference_tp}, using original image")
            data_ref.resliced_image = data_ref.image

        target_tps = time_points[1:]
        
        if len(target_tps) == 0:
            logger.info("No target timepoints to propagate to")
            return tp_input_data

        try:
            # Run registrations sequentially - RegistrationManager handles resource management
            for target_tp in target_tps:
                logger.info(f"Warping from reference tp {reference_tp} to target tp {target_tp}")
                
                data_target = tp_input_data[target_tp]

                # Run registration
                result = registration_manager.submit(
                    REGISTRATION_METHODS.RUN_REGISTRATION_AND_RESLICE,
                    img_fixed=data_target.image,
                    img_moving=data_ref.image,
                    img_to_reslice=data_ref.resliced_image,
                    mesh_to_reslice={},
                    options={}
                ).result()

                resliced_image = result.get('resliced_image', None)
                tp_input_data[target_tp].resliced_image = resliced_image
                
                logger.info(f"Completed registration for target tp {target_tp}")

        except Exception as e:
            logger.error(f"Registration failed: {e}", exc_info=True)
            raise

        return tp_input_data

    def get_strategy_name(self) -> str:
        return PropagationStrategyName.STAR