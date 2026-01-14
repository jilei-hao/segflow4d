from dataclasses import dataclass
from typing import Optional
from common.types.propagation_input import PropagationInput
from common.types.propagation_options import PropagationOptions
from common.types.propagation_strategy_data import PropagationStrategyData
from common.types.propagation_strategy_name import PropagationStrategyName
from common.types.tp_data import TPData
from common.types.tp_partition_input import TPPartitionInput
from propagation.propagation_strategy.propagation_strategy_factory import PropagationStrategyFactory
from propagation.tp_partition import TPPartition
from registration.registration_manager import RegistrationManager
from processing.image_processing import create_high_res_mask
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Add thread-local GPU synchronization lock
_gpu_lock = threading.Lock()

class PropagationPipeline:
    def __init__(self, input: PropagationInput):
        self._options = input.options
        self._input = input
        self._tp_partitions = None

        # initialize registration manager
        self._registration_manager = RegistrationManager(
            registration_backend=self._options.registration_backend,
            required_vram_mb=18000, # 18G
            vram_check_interval= 2.0 # seconds
        )


    def _validate_input(self):
        # provided time points are within the range of the 4D image and no overlap between partitions
        # provided segmentation images are compatible with the 4D image (dim, spacing, origin, direction)

        pass

    def _prepare_data(self):
        if self._input.image_4d is None:
            raise ValueError("4D image is not provided in the input.")
        
        self._tp_partitions = []
        for tp_partition_input in self._input.tp_input_groups:
            tp_partition = TPPartition(
                input=tp_partition_input,
                image_4d=self._input.image_4d,
                options=self._options
            )
            self._tp_partitions.append(tp_partition)
        
        self._input.image_4d = None  # free memory

    def run_low_res_propagation(self):
        pass


    def run_high_res_propagation(self):
        pass

    def _run_unidirectional_propagation(self, tp_data: dict[int, TPData], ref_input: TPPartitionInput, tp_list: list[int], options: PropagationOptions):
        # Add thread identification for better logging
        import threading
        thread_id = threading.get_ident()
        logger.info(f"[Thread {thread_id}] Running unidirectional propagation for time points: {tp_list}")

        # Synchronize GPU access between threads
        with _gpu_lock:
            # Force cleanup before starting
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        # prepare input data for propagation strategy
        tp_input_data = dict[int, PropagationStrategyData]()
        for tp in tp_list:
            crnt_tp_data = tp_data[tp]
            tp_input_data[tp] = PropagationStrategyData(image=crnt_tp_data.image_low_res)

        tp_ref = tp_list[0] # initialize the reference time point with mask to be propagated
        tp_input_data[tp_ref].resliced_image = tp_data[tp_ref].mask_low_res
        # Initialize variables to ensure they exist for cleanup
        propagated_data_lr = None
        propagated_data_hr = None
        tp_input_data_hr = None

        try:
            # create propagation strategy for low res propagation
            strategy_lr = PropagationStrategyFactory.create_propagation_strategy(PropagationStrategyName.SEQUENTIAL)

            # run low-res propagation for masks
            propagated_data_lr = strategy_lr.propagate(tp_input_data, options)

            # update propagated results back to tp_partition
            for tp in tp_list:
                resliced_mask = propagated_data_lr[tp].resliced_image
                if resliced_mask is None:
                    raise RuntimeError(f"Resliced mask for time point {tp} is None.")
                
                tp_data[tp].mask_low_res = resliced_mask

                high_res_mask = create_high_res_mask(ref_seg_image=ref_input.seg_ref, low_res_mask=resliced_mask)
                tp_data[tp].mask_high_res = high_res_mask
                
                if options.debug:
                    from utility.io.async_writer import async_writer
                    import os
                    if resliced_mask is not None:
                        async_writer.submit_image(
                            resliced_mask,
                            os.path.join(
                                options.debug_output_directory,
                                f"mask-lr_tp-{tp:03d}.nii.gz"
                            )
                        )
                    if high_res_mask is not None:
                        async_writer.submit_image(
                            high_res_mask,
                            os.path.join(
                                options.debug_output_directory,
                                f"mask-hr_tp-{tp:03d}.nii.gz"
                            )
                        )

            # Aggressive cleanup between stages
            tp_input_data.clear()
            propagated_data_lr.clear()
            strategy_lr = None
            
            # Force cleanup with GPU synchronization
            with _gpu_lock:
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            # create propagation strategy for high res propagation
            strategy_hr = PropagationStrategyFactory.create_propagation_strategy(PropagationStrategyName.STAR)

            # prepare input data for high res propagation
            tp_input_data_hr = dict[int, PropagationStrategyData]()
            for tp in tp_list:
                crnt_tp_data = tp_data[tp]
                tp_input_data_hr[tp] = PropagationStrategyData(image=crnt_tp_data.image)
                tp_input_data_hr[tp].mask = crnt_tp_data.mask_high_res
    
            tp_ref = tp_list[0] # initialize the reference time point with mask to be propagated
            tp_input_data_hr[tp_ref].resliced_image = ref_input.seg_ref

            # run high-res propagation for segmentations
            propagated_data_hr = strategy_hr.propagate(tp_input_data_hr, options)
            # update propagated results back to tp_partition
            for tp in tp_list:
                resliced_seg = propagated_data_hr[tp].resliced_image
                if resliced_seg is None:
                    raise RuntimeError(f"Resliced segmentation for time point {tp} is None.")
                
                crnt_tp_data = tp_data[tp]
                crnt_tp_data.segmentation = resliced_seg

                if options.debug:
                    from utility.io.async_writer import async_writer
                    import os
                    if resliced_seg is not None:
                        async_writer.submit_image(
                            resliced_seg,
                            os.path.join(
                                options.debug_output_directory,
                                f"seg-hr_tp-{tp:03d}.nii.gz"
                            )
                        )
                        
        finally:
            # Ensure cleanup happens even if errors occur
            import torch
            import gc
            
            # Clear Python references
            if 'tp_input_data' in locals():
                tp_input_data.clear()
            if propagated_data_lr is not None:
                propagated_data_lr.clear()
            if tp_input_data_hr is not None:
                tp_input_data_hr.clear()
            if propagated_data_hr is not None:
                propagated_data_hr.clear()
                
            # Final synchronized cleanup
            with _gpu_lock:
                # Force garbage collection
                gc.collect()
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            logger.debug(f"[Thread {thread_id}] Memory cleanup completed for timepoints {tp_list}")





    def _run_partition(self, tp_partition: TPPartition):
        # create an list contains all time points
        all_tps = tp_partition._input.tp_target.copy()
        all_tps.append(tp_partition._input.tp_ref)
        all_tps = list(set(all_tps))  # remove duplicates
        all_tps.sort()

        # divide into forward and backward time points
        forward_tps = [tp for tp in all_tps if tp >= tp_partition._input.tp_ref]
        backward_tps = [tp for tp in all_tps if tp <= tp_partition._input.tp_ref]

        # sort time points
        forward_tps.sort()
        backward_tps.sort(reverse=True)

        # prepare propagation tasks
        tasks = []
        
        # prepare forward propagation task
        if len(forward_tps) > 1:
            forward_tp_data = dict[int, TPData]()
            for tp in forward_tps:
                forward_tp_data[tp] = tp_partition._tp_data[tp]
            tasks.append(("forward", forward_tp_data, forward_tps))
        
        # prepare backward propagation task
        if len(backward_tps) > 1:
            backward_tp_data = dict[int, TPData]()
            for tp in backward_tps:
                backward_tp_data[tp] = tp_partition._tp_data[tp]
            tasks.append(("backward", backward_tp_data, backward_tps))

        # Check if we should run in parallel or sequential based on GPU memory
        use_parallel = len(tasks) == 2 and self._should_use_parallel_execution()
        
        if use_parallel:
            logger.info("Running forward and backward propagation in parallel")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # submit tasks
                future_to_direction = {}
                for direction, tp_data, tp_list in tasks:
                    future = executor.submit(
                        self._run_unidirectional_propagation, 
                        tp_data, 
                        tp_partition._input, 
                        tp_list, 
                        self._options
                    )
                    future_to_direction[future] = direction
                
                # wait for completion and handle any exceptions
                for future in as_completed(future_to_direction):
                    direction = future_to_direction[future]
                    try:
                        future.result()  # This will raise any exception that occurred
                        logger.info(f"{direction} propagation completed successfully")
                    except Exception as exc:
                        logger.error(f"{direction} propagation failed with exception: {exc}")
                        raise
        else:
            # Execute tasks sequentially to avoid memory issues
            for direction, tp_data, tp_list in tasks:
                logger.info(f"Running {direction} propagation sequentially")
                self._run_unidirectional_propagation(tp_data, tp_partition._input, tp_list, self._options)

    def _should_use_parallel_execution(self) -> bool:
        """Check if we have enough GPU memory for parallel execution"""
        import torch
        if not torch.cuda.is_available():
            return True  # CPU execution can handle parallel better
        
        # Check available VRAM
        gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        gpu_memory_free_mb = (torch.cuda.get_device_properties(0).total_memory - 
                             torch.cuda.memory_allocated()) / (1024**2)
        
        # Require at least 20GB total and 15GB free for parallel execution
        required_total_mb = 20000
        required_free_mb = 15000
        
        can_parallel = gpu_memory_mb > required_total_mb and gpu_memory_free_mb > required_free_mb
        
        if not can_parallel:
            logger.warning(f"Insufficient GPU memory for parallel execution. "
                          f"Total: {gpu_memory_mb:.0f}MB, Free: {gpu_memory_free_mb:.0f}MB. "
                          f"Falling back to sequential execution.")
        
        return can_parallel

    def run(self):
        self._validate_input()
        self._prepare_data()

        if self._tp_partitions is None:
            raise RuntimeError("No timepoint partitions available for processing.")

        for tp_partition in self._tp_partitions:
            self._run_partition(tp_partition)