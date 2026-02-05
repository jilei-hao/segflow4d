from dataclasses import dataclass
from typing import Optional
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
import numpy as np

@dataclass
class RegistrationOutput:
    '''
    A class representing the output of a registration process.
    '''
    affine_matrix: np.ndarray
    resliced_image: ImageWrapper
    resliced_segmentation_mesh: Optional[MeshWrapper]
    resliced_meshes: dict[str, MeshWrapper]
    warp_image: ImageWrapper