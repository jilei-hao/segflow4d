from dataclasses import dataclass
from .mesh_wrapper import MeshWrapper
from .image_wrapper import ImageWrapper
from typing import Optional

@dataclass
class TPPartitionInput:
    '''
    A class representing group of inputs for running propagation on one partition of all the timepoints.
    '''
    seg_ref: ImageWrapper # used for full resolution propagation
    additional_meshes_ref: Optional[dict[str, MeshWrapper]] # additional mesh representations of structures in the segmentation reference image
    tp_ref: int # timepoint index of the reference image used for segmentation propagation
    tp_target: list[int] # list of timepoint indices of the target images for segmentation propagation