from dataclasses import dataclass
from common.types.tp_image_group import TPImageGroup

@dataclass
class TPPartitionOutput:
    '''
    A class representing group of outputs (all data other than the input are output) after running propagation on one partition of all the timepoints.
    '''
    pass