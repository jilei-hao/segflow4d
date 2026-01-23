from typing import Optional
from common.types.image_wrapper import ImageWrapper
from common.types.propagation_input import PropagationInput
from common.types.propagation_options import PropagationOptions
from common.types.propagation_strategy_data import PropagationStrategyData
from common.types.propagation_strategy_name import PropagationStrategyName
from common.types.tp_data import TPData
from propagation.tp_partition_input import TPPartitionInput
from propagation.propagation_strategy.propagation_strategy_factory import PropagationStrategyFactory
from propagation.tp_partition import TPPartition
from registration.registration_manager import RegistrationManager
from utility.image_helper.image_helper_factory import create_image_helper
from processing.image_processing import create_high_res_mask
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import torch

logger = logging.getLogger(__name__)


class PropagationPipeline:
    def __init__(self, input: PropagationInput):
        self._options = input.options
        self._input = input
        self._tp_partitions = None

        # initialize registration manager
        self._registration_manager = RegistrationManager(
            registration_backend=self._options.registration_backend,
            required_vram_mb=18000,  # 18G
            vram_check_interval=0.5  # seconds
        )

    def _validate_input(self):
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

    def _run_unidirectional_propagation(self, tp_data: dict[int, TPData], ref_input: TPPartitionInput, tp_list: list[int], options: PropagationOptions) -> dict[int, TPData]:
        thread_id = threading.get_ident()
        logger.info(f"[Thread {thread_id}] Running unidirectional propagation for time points: {tp_list}")

        # prepare input data for propagation strategy
        # IMPORTANT: Deep copy ImageWrapper to ensure thread isolation
        tp_input_data = dict[int, PropagationStrategyData]()
        for tp in tp_list:
            crnt_tp_data = tp_data[tp]
            tp_input_data[tp] = PropagationStrategyData(
                image=crnt_tp_data.image_low_res.deepcopy() if crnt_tp_data.image_low_res else None
            )

        tp_ref = tp_list[0]
        mask_low_res = tp_data[tp_ref].mask_low_res
        if mask_low_res is not None:
            tp_input_data[tp_ref].resliced_image = mask_low_res.deepcopy()
        
        # Initialize variables
        propagated_data_lr = None
        propagated_data_hr = None
        tp_input_data_hr = None
        result = dict[int, TPData]()

        try:
            # ===== LOW RES PROPAGATION =====
            logger.debug(f"[Thread {thread_id}] Starting low-res propagation")
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
                            os.path.join(options.debug_output_directory, f"mask-lr_tp-{tp:03d}.nii.gz")
                        )
                    if high_res_mask is not None:
                        async_writer.submit_image(
                            high_res_mask,
                            os.path.join(options.debug_output_directory, f"mask-hr_tp-{tp:03d}.nii.gz")
                        )

            logger.debug(f"[Thread {thread_id}] Low-res propagation completed")
            
            # Clear references for next stage
            tp_input_data = None
            propagated_data_lr = None

            # ===== HIGH RES PROPAGATION =====
            logger.debug(f"[Thread {thread_id}] Starting high-res propagation")
            strategy_hr = PropagationStrategyFactory.create_propagation_strategy(PropagationStrategyName.STAR)

            # prepare input data for high res propagation
            # IMPORTANT: Deep copy ImageWrapper to ensure thread isolation
            tp_input_data_hr = dict[int, PropagationStrategyData]()
            for tp in tp_list:
                crnt_tp_data = tp_data[tp]
                tp_input_data_hr[tp] = PropagationStrategyData(
                    image=crnt_tp_data.image.deepcopy() if crnt_tp_data.image else None,
                    mask=crnt_tp_data.mask_high_res.deepcopy() if crnt_tp_data.mask_high_res else None
                )

            tp_ref = tp_list[0]
            tp_input_data_hr[tp_ref].resliced_image = ref_input.seg_ref.deepcopy()
            tp_input_data_hr[tp_ref].segmentation_mesh = ref_input.seg_mesh_ref.deepcopy() if ref_input.seg_mesh_ref else None

            # run high-res propagation for segmentations
            propagated_data_hr = strategy_hr.propagate(tp_input_data_hr, options)
            
            # update propagated results back to tp_partition
            if options.debug:
                from utility.io.async_writer import async_writer
                import os

                for tp in tp_list:
                    resliced_seg = propagated_data_hr[tp].resliced_image
                    if resliced_seg is None:
                        raise RuntimeError(f"Resliced segmentation for time point {tp} is None.")

                    async_writer.submit_image(
                        resliced_seg,
                        os.path.join(options.debug_output_directory, f"seg-hr_tp-{tp:03d}.nii.gz")
                    )


            for tp in tp_list:
                logger.debug(f"[Thread {thread_id}] Processing time point {tp}")
                if tp == tp_ref:
                    result[tp] = tp_data[tp]  # original data for reference time point
                    continue

                result[tp] = tp_data[tp].deepcopy()
                resliced_image = propagated_data_hr[tp].resliced_image
                if resliced_image is not None:
                    result[tp].segmentation = resliced_image

            logger.info(f"[Thread {thread_id}] Propagation completed successfully for timepoints {tp_list}")
                        
        except Exception as e:
            logger.error(f"[Thread {thread_id}] Error during propagation: {str(e)}", exc_info=True)
            raise
            
        finally:
            logger.debug(f"[Thread {thread_id}] Cleaning up propagation resources")
            
            # Clear all local references
            if propagated_data_lr is not None:
                propagated_data_lr.clear() if hasattr(propagated_data_lr, 'clear') else None
            if tp_input_data_hr is not None:
                tp_input_data_hr.clear() if hasattr(tp_input_data_hr, 'clear') else None
            if propagated_data_hr is not None:
                propagated_data_hr.clear() if hasattr(propagated_data_hr, 'clear') else None
            
            logger.debug(f"[Thread {thread_id}] Cleanup completed")

        return result


    def _run_partition(self, tp_partition: TPPartition):
        # create a list containing all time points
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

        # results container
        results = dict[str, dict[int, TPData]]()
        
        # prepare forward propagation task
        if len(forward_tps) > 1:
            forward_tp_data = dict[int, TPData]()
            for tp in forward_tps:
                forward_tp_data[tp] = tp_partition._tp_data[tp].deepcopy()
            tasks.append(("forward", forward_tp_data, forward_tps))
        
        # prepare backward propagation task
        if len(backward_tps) > 1:
            backward_tp_data = dict[int, TPData]()
            for tp in backward_tps:
                backward_tp_data[tp] = tp_partition._tp_data[tp].deepcopy()
            tasks.append(("backward", backward_tp_data, backward_tps))

        # execute tasks in parallel if we have both forward and backward
        if len(tasks) > 1:
            logger.info("Running forward and backward propagation in parallel")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_direction = {}
                for direction, tp_data, tp_list in tasks:
                    future = executor.submit(
                        self._run_unidirectional_propagation, 
                        tp_data, 
                        tp_partition._input.deepcopy(),
                        tp_list, 
                        self._options
                    )
                    future_to_direction[future] = direction
                
                # wait for completion and handle any exceptions
                for future in as_completed(future_to_direction):
                    direction = future_to_direction[future]
                    try:
                        directional_result = future.result()
                        results[direction] = directional_result
                        logger.info(f"{direction} propagation completed successfully")
                    except Exception as exc:
                        logger.error(f"{direction} propagation failed with exception: {exc}", exc_info=True)
                        sys.exit(1)
                        
        elif len(tasks):
            # execute single task sequentially
            direction, tp_data, tp_list = tasks[0]
            logger.info(f"Running {direction} propagation only")
            results[direction] = self._run_unidirectional_propagation(tp_data, tp_partition._input.deepcopy(), tp_list, self._options)
        else:
            logger.info("No propagation tasks to execute (only reference timepoint)")

        # combine directional results
        result = dict[int, TPData]()
        for direction, directional_result in results.items():
            for tp, tp_data in directional_result.items():
                result[tp] = tp_partition._tp_data[tp]  # original data
                if tp_data.segmentation is not None:
                    result[tp].segmentation = tp_data.segmentation

        return result
    

    def write_results_to_disk(self, all_tp_output: dict[int, TPData]):
        from utility.io.async_writer import async_writer
        import os

        if self._tp_partitions is None:
            raise RuntimeError("No timepoint partitions available for writing results.")

        # create output directories
        os.makedirs(self._options.output_directory, exist_ok=True)
        seg_mesh_output_dir = os.path.join(self._options.output_directory, "segmentation_meshes")
        os.makedirs(seg_mesh_output_dir, exist_ok=True)

        tp_images = list[ImageWrapper]()
        tp_segmentations = list[ImageWrapper]()

        for tp in sorted(all_tp_output.keys()):
            tp_data = all_tp_output[tp]
            tp_images.append(tp_data.image)
            if tp_data.segmentation is not None:
                tp_segmentations.append(tp_data.segmentation)
            else:
                logger.error(f"Segmentation for time point {tp} is None, cannot write to disk.")
                raise RuntimeError(f"Segmentation for time point {tp} is None.")

        # Write images
        ih = create_image_helper()
        image_4d = ih.create_4d_image(tp_images)
        seg_4d = ih.create_4d_image(tp_segmentations)

        async_writer.submit_image(
            image_4d,
            os.path.join(self._options.output_directory, "image-4d.nii.gz")
        )

        async_writer.submit_image(
            seg_4d,
            os.path.join(self._options.output_directory, "seg-4d.nii.gz")
        )

        # Write segmentation meshes
        for tp in sorted(all_tp_output.keys()):
            tp_data = all_tp_output[tp]
            if tp_data.segmentation_mesh is not None:
                mesh_output_path = os.path.join(seg_mesh_output_dir, f"seg-mesh_tp-{tp:03d}.vtp")
                async_writer.submit_mesh(
                    tp_data.segmentation_mesh,
                    mesh_output_path
                )
            else:
                logger.warning(f"Segmentation mesh for time point {tp} is None, skipping mesh write.")




    def run(self):
        self._validate_input()
        self._prepare_data()

        if self._tp_partitions is None:
            raise RuntimeError("No timepoint partitions available for processing.")
        
        partition_results = list[dict[int, TPData]]()

        for tp_partition in self._tp_partitions:
            crnt_result = self._run_partition(tp_partition)
            partition_results.append(crnt_result)

        # create all tp output
        all_tp_output = dict[int, TPData]()
        for crnt_result in partition_results:
            for tp, tp_data in crnt_result.items():
                all_tp_output[tp] = tp_data

        if self._options.write_result_to_disk:
            self.write_results_to_disk(all_tp_output)


        logger.info("Propagation pipeline completed successfully.")