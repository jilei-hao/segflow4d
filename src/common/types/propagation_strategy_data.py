from dataclasses import dataclass
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from typing import Optional


@dataclass
class PropagationStrategyData():
    image: Optional[ImageWrapper] = None
    mask: Optional[ImageWrapper] = None
    resliced_image: Optional[ImageWrapper] = None
    resliced_meshes: Optional[dict[str, MeshWrapper]] = None
