from abc import ABC, abstractmethod
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from typing import Any


class AbstractSegmentationMeshGenerator(ABC):
    @abstractmethod
    def set_inputs(self, image: ImageWrapper, options: Any):
        """
        Set the input image and options for segmentation mesh generation.

        Args:
            image: The input image to be segmented.
            options: A dictionary of options for segmentation.
        """
        pass


    @abstractmethod
    def generate_mesh(self) -> MeshWrapper:
        """
        Generate the segmentation mesh based on the input image and options.

        Returns:
            The generated segmentation mesh.
        """
        pass