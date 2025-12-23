from dataclasses import dataclass
from .tp_partition_input import TPPartitionInputGroup
from .tp_partition_output import TPPartitionOutputGroup
from typing import Optional

@dataclass
class TPPartition:
    '''
    A class representing one partition of all the timepoints for propagation.
    '''
    input_group: TPPartitionInputGroup
    output_group: Optional[dict[int, TPPartitionOutputGroup]] # key: all timepoint index in this partition (ref + targets)