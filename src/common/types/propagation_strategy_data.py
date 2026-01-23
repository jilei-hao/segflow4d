from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from typing import Optional


class PropagationStrategyData:
    """
    Data structure for propagation strategy containing image and mesh data.
    """
    def __init__(
        self,
        image: Optional[ImageWrapper] = None,
        mask: Optional[ImageWrapper] = None,
        resliced_image: Optional[ImageWrapper] = None,
        resliced_meshes: Optional[dict[str, MeshWrapper]] = None,
        segmentation_mesh: Optional[MeshWrapper] = None
    ):
        self.image = image
        self.mask = mask
        self.resliced_image = resliced_image
        self.resliced_meshes = resliced_meshes
        self.segmentation_mesh = segmentation_mesh
    
    def deepcopy(self) -> 'PropagationStrategyData':
        """Create a deep copy of PropagationStrategyData with independent image and mesh data."""
        # Deep copy ImageWrapper objects
        def _deepcopy_image(img: Optional[ImageWrapper]) -> Optional[ImageWrapper]:
            if img is None:
                return None
            return img.deepcopy()
        
        # Deep copy MeshWrapper objects
        def _deepcopy_mesh(mesh: Optional[MeshWrapper]) -> Optional[MeshWrapper]:
            if mesh is None:
                return None
            return mesh.deepcopy() if hasattr(mesh, 'deepcopy') else mesh
        
        # Deep copy meshes dictionary
        resliced_meshes_copy = None
        if self.resliced_meshes is not None:
            resliced_meshes_copy = {
                key: copied_mesh
                for key, mesh in self.resliced_meshes.items()
                if mesh is not None and (copied_mesh := _deepcopy_mesh(mesh)) is not None
            }
        
        return PropagationStrategyData(
            image=_deepcopy_image(self.image),
            mask=_deepcopy_image(self.mask),
            resliced_image=_deepcopy_image(self.resliced_image),
            resliced_meshes=resliced_meshes_copy,
            segmentation_mesh=_deepcopy_mesh(self.segmentation_mesh)
        )
    
    def clear(self):
        """Clear all data references."""
        self.image = None
        self.mask = None
        self.resliced_image = None
        self.resliced_meshes = None
        self.segmentation_mesh = None