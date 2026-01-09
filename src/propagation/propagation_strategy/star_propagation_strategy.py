import torch
import gc
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

        try:
            for i in range(1, len(time_points)):
                logger.info(f"-- Warping from reference time point {reference_tp} to time point {time_points[i]} --")
                data_ref = tp_input_data[reference_tp]
                data_target = tp_input_data[time_points[i]]

                # Ensure reference timepoint has resliced_image
                if data_ref.resliced_image is None:
                    logger.warning(f"No resliced_image for reference timepoint {reference_tp}, using original image")
                    data_ref.resliced_image = data_ref.image

                # run registration from reference to target
                result = registration_manager.submit(
                    REGISTRATION_METHODS.RUN_REGISTRATION_AND_RESLICE,
                    img_fixed=data_target.image,
                    img_moving=data_ref.image,
                    img_to_reslice=data_ref.resliced_image,
                    mesh_to_reslice={},
                    options={}
                ).result()

                resliced_image = result.get('resliced_image', None)
                data_target.resliced_image = resliced_image

                tp_input_data[time_points[i]] = data_target
                
                # Aggressive cleanup after each registration (similar to SequentialPropagationStrategy pattern)
                del result
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                logger.debug(f"Completed registration and cleanup for timepoint {time_points[i]}")

        finally:
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return tp_input_data

    def get_strategy_name(self) -> str:
        return PropagationStrategyName.STAR