from dataclasses import dataclass
from common.types.tp_image_group import TPImageGroup

@dataclass
class TPData:
    '''
    A class representing data used during propagation at a specific timepoint.
    '''
    image_group: TPImageGroup
    mask_group: TPImageGroup
    segmentation_group: TPImageGroup