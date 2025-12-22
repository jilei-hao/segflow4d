from common.types.propagation_input import PropagationInput
from common.types.tp_partition_input_group import TPPartitionInputGroup
from common.types.tp_data import TPData
import processing.image_processing as img_proc


class PropagationPipeline:
    def __init__(self, input: PropagationInput):
        self._options = input.options
        self._input_image4d = input.image_4d.get_data()
        self._tp_input_groups = input.tp_input_groups
        self._tp_data = {}

    def _validate_input(self):
        # provided time points are within the range of the 4D image and no overlap between partitions
        # provided segmentation images are compatible with the 4D image (dim, spacing, origin, direction)

        pass

    def _prepare_data(self):
        # prepare timepoint images
        pass


    def _propagate(self):
        pass