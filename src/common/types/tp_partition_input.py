from dataclasses import dataclass
from .mesh_wrapper import MeshWrapper
from .image_wrapper import ImageWrapper
from typing import Optional
import copy

@dataclass
class TPPartitionInput:
    '''
    A class representing group of inputs for running propagation on one partition of all the timepoints.
    '''
    seg_ref: ImageWrapper # used for full resolution propagation
    additional_meshes_ref: Optional[dict[str, MeshWrapper]] # additional mesh representations of structures in the segmentation reference image
    tp_ref: int # timepoint index of the reference image used for segmentation propagation
    tp_target: list[int] # list of timepoint indices of the target images for segmentation propagation
    
    def deepcopy(self) -> 'TPPartitionInput':
        # Deep copy the additional meshes dictionary and its contents
        if self.additional_meshes_ref:
            additional_meshes_copy = {
                key: mesh.deepcopy() if hasattr(mesh, 'deepcopy') else copy.deepcopy(mesh)
                for key, mesh in self.additional_meshes_ref.items()
            }
        else:
            additional_meshes_copy = None
        
        return TPPartitionInput(
            seg_ref=self.seg_ref.deepcopy(),
            additional_meshes_ref=additional_meshes_copy,
            tp_ref=self.tp_ref,
            tp_target=self.tp_target.copy()  # list.copy() is also shallow, but ints are immutable so OK
        )