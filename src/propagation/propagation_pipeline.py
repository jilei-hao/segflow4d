from typing import Optional
from common.types.propagation_input import PropagationInput
from propagation.tp_partition import TPPartition


class PropagationPipeline:
    def __init__(self, input: PropagationInput):
        self._options = input.options
        self._input = input
        self._tp_partitions = None

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


    def _run_unidirectional_propagation(self, tp_list: list[int]):
        pass


    def _run_partition(self, tp_partition: TPPartition):
        pass


    def run(self):
        self._validate_input()
        self._prepare_data()

        if self._tp_partitions is None:
            raise RuntimeError("No timepoint partitions available for processing.")

        for tp_partition in self._tp_partitions:
            self._run_partition(tp_partition)