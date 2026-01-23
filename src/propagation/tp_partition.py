from common.types.interpolation_type import InterpolationType
from common.types.propagation_options import PropagationOptions
from common.types.tp_data import TPData
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from utility.image_helper.image_helper_factory import create_image_helper
from processing.image_processing import create_reference_mask, create_high_res_mask
from propagation.tp_partition_input import TPPartitionInput
from logging import getLogger
from utility.io.async_writer import async_writer
import os

logger = getLogger(__name__)

class TPPartition:
    '''
    A class representing one partition of all the timepoints for propagation.
    '''
    def __init__(self, input: TPPartitionInput, image_4d: ImageWrapper, options: PropagationOptions):
        self._input = input
        self._options = options
        self._tp_data = self._initialize_tp_data(image_4d)


    def _initialize_tp_data(self, image_4d: ImageWrapper) -> dict[int, TPData]:
        tp_data_dict = {}
        all_timepoints = [self._input.tp_ref] + self._input.tp_target
        # sort timepoints
        all_timepoints.sort()

        ih = create_image_helper()

        for tp in all_timepoints:
            logger.info(f"Extracting timepoint {tp} from 4D image for TPPartition")
            tp_image = ih.extract_timepoint_image(image_4d, tp)

            logger.info(f"Resampling timepoint {tp} image to low resolution for TPPartition")
            tp_image_low_res = ih.resample(tp_image, resample_factor=self._options.lowres_resample_factor, interpolation=InterpolationType.LINEAR)

            tp_data_dict[tp] = TPData(image=tp_image, image_low_res=tp_image_low_res)

            if self._options.debug:
                async_writer.submit_image(tp_image, os.path.join(self._options.debug_output_directory, f"img_tp-{tp:03d}.nii.gz"))
                async_writer.submit_image(tp_image_low_res, os.path.join(self._options.debug_output_directory, f"img-lr_tp-{tp:03d}.nii.gz"))

        tp_data_dict[self._input.tp_ref].segmentation = self._input.seg_ref

        logger.info(f"Creating reference mask for timepoint {self._input.tp_ref} in TPPartition")
        mask_ref_lr = create_reference_mask(tp_data_dict[self._input.tp_ref].segmentation, resample_factor=self._options.lowres_resample_factor, dilation_radius=self._options.dilation_radius)
        tp_data_dict[self._input.tp_ref].mask_low_res = mask_ref_lr
        tp_data_dict[self._input.tp_ref].mask_high_res = create_high_res_mask(ref_seg_image=self._input.seg_ref, low_res_mask=mask_ref_lr)

        if self._options.debug:
            async_writer.submit_image(tp_data_dict[self._input.tp_ref].mask_low_res, os.path.join(self._options.debug_output_directory, f"mask-ref-lr_tp-{self._input.tp_ref:03d}.nii.gz"))
            async_writer.submit_image(tp_data_dict[self._input.tp_ref].mask_high_res, os.path.join(self._options.debug_output_directory, f"mask-ref-hr_tp-{self._input.tp_ref:03d}.nii.gz"))

        tp_data_dict[self._input.tp_ref].segmentation_mesh = None  # Could be set if needed
        tp_data_dict[self._input.tp_ref].additional_meshes = self._input.additional_meshes_ref

        return tp_data_dict