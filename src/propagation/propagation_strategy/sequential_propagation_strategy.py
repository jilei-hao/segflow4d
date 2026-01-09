from propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
from common.types.propagation_strategy_data import PropagationStrategyData
from registration.registration_manager import RegistrationManager
from common.types.registration_methods import REGISTRATION_METHODS
from common.types.propagation_strategy_name import PropagationStrategyName
from logging import getLogger

logger = getLogger(__name__)

class SequentialPropagationStrategy(AbstractPropagationStrategy):
    def propagate(self, tp_input_data: dict[int, PropagationStrategyData], options) -> dict[int, PropagationStrategyData]:
        # get all time points
        time_points = list(tp_input_data.keys())
        logger.info(f"SequentialPropagationStrategy: propagating through time points {time_points}")

        registration_manager = RegistrationManager.get_instance()

        # propagate sequentially from the first time point to the last
        for i in range(0, len(time_points) - 1):
            logger.info(f"-- Warping from time point {time_points[i]} to {time_points[i + 1]} --")
            data_crnt = tp_input_data[time_points[i]]
            data_next = tp_input_data[time_points[i + 1]]

            # Ensure current timepoint has resliced_image
            if data_crnt.resliced_image is None:
                logger.warning(f"No resliced_image for timepoint {time_points[i]}, using original image")
                data_crnt.resliced_image = data_crnt.image

            # run registration from current to next
            result = registration_manager.submit(
                REGISTRATION_METHODS.RUN_REGISTRATION_AND_RESLICE,
                img_fixed=data_next.image,
                img_moving=data_crnt.image,
                img_to_reslice=data_crnt.resliced_image,
                mesh_to_reslice={},
                options={}
            ).result()

            resliced_image = result.get('resliced_image', None)
            data_next.resliced_image = resliced_image

            tp_input_data[time_points[i + 1]] = data_next

        return tp_input_data

    def get_strategy_name(self) -> str:
        return PropagationStrategyName.SEQUENTIAL