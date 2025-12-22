from dataclasses import dataclass
from .image_wrapper import ImageWrapper

@dataclass
class TPImageGroup:
    '''
    A class representing a group of images objects in different resolutions representing
    the same structure used during propagation at a specific timepoint.
    '''
    image_fullres: ImageWrapper # used for full resolution propagation
    image_lowres: ImageWrapper # used for downsampled propagation