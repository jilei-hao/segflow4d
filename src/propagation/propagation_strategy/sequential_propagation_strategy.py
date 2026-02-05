from common.types.propagation_options import PropagationOptions
from propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
from common.types.tp_data import TPData
from registration.registration_manager import RegistrationManager
from common.types.registration_methods import REGISTRATION_METHODS
from common.types.propagation_strategy_name import PropagationStrategyName
from logging import getLogger

logger = getLogger(__name__)

class SequentialPropagationStrategy(AbstractPropagationStrategy):
    def propagate(self, tp_input_data: dict[int, TPData], options: PropagationOptions) -> dict[int, TPData]:
        """
        Propagate sequentially but batch jobs where possible.
        
        For true sequential propagation (where each step depends on the previous),
        we can still parallelize across independent chains.
        """
        registration_manager = RegistrationManager.get_instance()
        
        tp_list = list(tp_input_data.keys())
        logger.info(f"SequentialPropagationStrategy: propagating through time points {tp_list}")
        
        # For sequential, we need result from step N before step N+1
        # But we can still submit the NEXT job before waiting for current to complete
        # using a sliding window approach
        
        prev_future = None
        prev_tp = None
        
        for i, tp in enumerate(tp_list[1:], 1):
            src_tp = tp_list[i - 1]
            
            # Wait for previous result if exists (needed for true sequential dependency)
            if prev_future is not None and prev_tp is not None:
                result = prev_future.result()  # TPData object
                logger.info(f"Completed registration for tp {prev_tp}")
                tp_input_data[prev_tp].resliced_image = result.resliced_image
                tp_input_data[prev_tp].segmentation_mesh = result.resliced_segmentation_mesh
            logger.info(f"-- Warping from time point {src_tp} to {tp} --")
            
            # Submit next job immediately (don't wait)
            future = registration_manager.submit(
                'run_registration_and_reslice',
                img_fixed=tp_input_data[tp].image,
                img_moving=tp_input_data[src_tp].image,
                img_to_reslice=tp_input_data[src_tp].resliced_image,
                mesh_to_reslice=tp_input_data[src_tp].segmentation_mesh,
                options=options.registration_backend_options,
                mask_fixed=tp_input_data[tp].mask,
                mask_moving=tp_input_data[src_tp].mask
            )
            
            prev_future = future
            prev_tp = tp
        
        # Wait for final result
        if prev_future is not None and prev_tp is not None:
            result = prev_future.result()  # TPData object
            logger.info(f"Completed registration for tp {prev_tp}")
            tp_input_data[prev_tp].resliced_image = result.resliced_image
            tp_input_data[prev_tp].warp_image = result.warp_image
        
        return tp_input_data

    def get_strategy_name(self) -> str:
        return PropagationStrategyName.SEQUENTIAL