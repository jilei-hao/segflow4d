from common.types.propagation_options import PropagationOptions
from propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
from common.types.propagation_strategy_data import PropagationStrategyData
from registration.registration_manager import RegistrationManager
from common.types.registration_methods import REGISTRATION_METHODS
from common.types.propagation_strategy_name import PropagationStrategyName
from logging import getLogger

logger = getLogger(__name__)

class StarPropagationStrategy(AbstractPropagationStrategy):
    def propagate(self, tp_input_data: dict[int, PropagationStrategyData], options: PropagationOptions) -> dict[int, PropagationStrategyData]:
        """Propagate using star pattern - submit all jobs in parallel."""
        registration_manager = RegistrationManager.get_instance()
        
        tp_list = list(tp_input_data.keys())
        ref_tp = tp_list[0]
        target_tps = tp_list[1:]
        
        logger.info(f"StarPropagationStrategy: propagating through time points {tp_list}")
        
        # Submit ALL jobs first (non-blocking)
        futures = {}
        for target_tp in target_tps:
            logger.info(f"Submitting registration: reference tp {ref_tp} to target tp {target_tp}")
            
            future = registration_manager.submit(
                REGISTRATION_METHODS.RUN_REGISTRATION_AND_RESLICE,
                img_fixed=tp_input_data[target_tp].image,
                img_moving=tp_input_data[ref_tp].image,
                img_to_reslice=tp_input_data[ref_tp].resliced_image,
                mesh_to_reslice=tp_input_data[ref_tp].segmentation_mesh,
                options=options.registration_backend_options,
                mask_fixed=tp_input_data[target_tp].mask,
                mask_moving=tp_input_data[ref_tp].mask
            )
            futures[target_tp] = future
        
        logger.info(f"Submitted {len(futures)} registration jobs to queue")
        
        # Now collect results (this allows parallel execution)
        results = {}
        for target_tp, future in futures.items():
            try:
                result = future.result()  # Now we wait
                logger.info(f"Completed registration for target tp {target_tp}")
                results[target_tp] = result
            except Exception as e:
                logger.error(f"Registration failed for tp {target_tp}: {e}")
                raise
        
        # Process results
        for target_tp, result in results.items():
            tp_input_data[target_tp].resliced_image = result['resliced_image']
            tp_input_data[target_tp].segmentation_mesh = result['resliced_segmentation_mesh']
    
        return tp_input_data

    def get_strategy_name(self) -> str:
        return PropagationStrategyName.STAR