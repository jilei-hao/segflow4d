from dataclasses import dataclass
from typing import Optional

from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper

@dataclass
class TPData:
    '''
    A class representing the data for one timepoint.
    '''
    # result objects
    image: ImageWrapper
    image_low_res: Optional[ImageWrapper] = None
    segmentation: Optional[ImageWrapper] = None
    segmentation_mesh: Optional[MeshWrapper] = None
    mask_low_res: Optional[ImageWrapper] = None
    mask_high_res: Optional[ImageWrapper] = None
    additional_meshes: Optional[dict[str, MeshWrapper]] = None  # Replace 'any' with actual mesh type if available

    # transformation objects
    affine_from_prev: Optional[object] = None  # Replace 'object' with actual transformation type if available
    affine_from_ref: Optional[object] = None
    deformable_from_ref: Optional[object] = None
    deformable_from_ref_low_res: Optional[object] = None

