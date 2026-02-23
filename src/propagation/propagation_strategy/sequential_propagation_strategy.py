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
        Propagate sequentially - each step waits for the previous to complete.
        """
        registration_manager = RegistrationManager.get_instance()
        
        tp_list = list(tp_input_data.keys())
        logger.info(f"SequentialPropagationStrategy: propagating through time points {tp_list}")
        
        # Initialize seed timepoint's resliced_image if not already set
        seed_tp = tp_list[0]
        if tp_input_data[seed_tp].resliced_image is None:
            logger.info(f"Initializing resliced_image for seed timepoint {seed_tp}")
            tp_input_data[seed_tp].resliced_image = tp_input_data[seed_tp].image
        
        # Sequential propagation: process one timepoint at a time
        for i in range(1, len(tp_list)):
            src_tp = tp_list[i - 1]
            tgt_tp = tp_list[i]
            
            logger.info(f"-- Propagating from timepoint {src_tp} to {tgt_tp} --")
            logger.debug(f"Source tp {src_tp}: resliced_image={tp_input_data[src_tp].resliced_image is not None}, "
                        f"segmentation_mesh={tp_input_data[src_tp].segmentation_mesh is not None}")
            
            # Submit job
            future = registration_manager.submit(
                'run_registration_and_reslice',
                img_fixed=tp_input_data[tgt_tp].image,
                img_moving=tp_input_data[src_tp].image,
                img_to_reslice=tp_input_data[src_tp].resliced_image,
                mesh_to_reslice=tp_input_data[src_tp].segmentation_mesh,
                options=options,
                mask_fixed=tp_input_data[tgt_tp].mask,
                mask_moving=tp_input_data[src_tp].mask
            )
            
            # Wait for result before continuing
            logger.info(f"Waiting for registration to complete for timepoint {tgt_tp}...")
            result = future.result()
            
            # Update target timepoint with results
            logger.info(f"Registration completed for timepoint {tgt_tp}")
            logger.debug(f"Result: resliced_image={result.resliced_image is not None}, "
                        f"resliced_segmentation_mesh={result.resliced_segmentation_mesh is not None}")
            
            tp_input_data[tgt_tp].resliced_image = result.resliced_image
            tp_input_data[tgt_tp].segmentation_mesh = result.resliced_segmentation_mesh
            tp_input_data[tgt_tp].warp_image = result.warp_image
        
        logger.info("Sequential propagation completed")
        return tp_input_data

    def get_strategy_name(self) -> str:
        return PropagationStrategyName.SEQUENTIAL