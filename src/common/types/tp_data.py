from typing import Optional
import copy
import numpy as np

from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper

class TPData:
    '''
    A class representing the data for one timepoint.
    
    This class consolidates data from:
    - Original TPData (timepoint-level data)
    - RegistrationOutput (registration results)
    - PropagationStrategyData (propagation intermediate data)
    '''
    def __init__(
        self,
        image: Optional[ImageWrapper] = None,
        image_low_res: Optional[ImageWrapper] = None,
        segmentation: Optional[ImageWrapper] = None,
        segmentation_mesh: Optional[MeshWrapper] = None,
        mask: Optional[ImageWrapper] = None,
        mask_low_res: Optional[ImageWrapper] = None,
        mask_high_res: Optional[ImageWrapper] = None,
        additional_meshes: Optional[dict[str, MeshWrapper]] = None,
        # Registration output fields
        affine_matrix: Optional[np.ndarray] = None,
        resliced_image: Optional[ImageWrapper] = None,
        resliced_segmentation_mesh: Optional[MeshWrapper] = None,
        resliced_meshes: Optional[dict[str, MeshWrapper]] = None,
        warp_image: Optional[ImageWrapper] = None,
        # Transformation fields
        affine_from_prev: Optional[object] = None,
        affine_from_ref: Optional[object] = None,
        deformable_from_ref: Optional[object] = None,
        deformable_from_ref_low_res: Optional[object] = None
    ):
        # Core image and segmentation data
        self.image = image
        self.image_low_res = image_low_res
        self.segmentation = segmentation
        self.segmentation_mesh = segmentation_mesh
        
        # Mask data
        self.mask = mask
        self.mask_low_res = mask_low_res
        self.mask_high_res = mask_high_res
        
        # Additional meshes
        self.additional_meshes = additional_meshes
        
        # Registration output fields
        self.affine_matrix = affine_matrix
        self.resliced_image = resliced_image
        self.resliced_segmentation_mesh = resliced_segmentation_mesh
        self.resliced_meshes = resliced_meshes
        self.warp_image = warp_image

        # Transformation objects
        self.affine_from_prev = affine_from_prev
        self.affine_from_ref = affine_from_ref
        self.deformable_from_ref = deformable_from_ref
        self.deformable_from_ref_low_res = deformable_from_ref_low_res

    def deepcopy(self) -> 'TPData':
        """Create a deep copy of TPData with copied ImageWrapper and MeshWrapper objects."""
        return TPData(
            image=self.image.deepcopy() if self.image else None,
            image_low_res=self.image_low_res.deepcopy() if self.image_low_res else None,
            segmentation=self.segmentation.deepcopy() if self.segmentation else None,
            segmentation_mesh=self.segmentation_mesh.deepcopy() if self.segmentation_mesh else None,
            mask=self.mask.deepcopy() if self.mask else None,
            mask_low_res=self.mask_low_res.deepcopy() if self.mask_low_res else None,
            mask_high_res=self.mask_high_res.deepcopy() if self.mask_high_res else None,
            additional_meshes={k: v.deepcopy() for k, v in self.additional_meshes.items()} if self.additional_meshes else None,
            affine_matrix=self.affine_matrix.copy() if self.affine_matrix is not None else None,
            resliced_image=self.resliced_image.deepcopy() if self.resliced_image else None,
            resliced_segmentation_mesh=self.resliced_segmentation_mesh.deepcopy() if self.resliced_segmentation_mesh else None,
            resliced_meshes={k: v.deepcopy() for k, v in self.resliced_meshes.items()} if self.resliced_meshes else None,
            warp_image=self.warp_image.deepcopy() if self.warp_image else None,
            affine_from_prev=copy.deepcopy(self.affine_from_prev) if self.affine_from_prev else None,
            affine_from_ref=copy.deepcopy(self.affine_from_ref) if self.affine_from_ref else None,
            deformable_from_ref=copy.deepcopy(self.deformable_from_ref) if self.deformable_from_ref else None,
            deformable_from_ref_low_res=copy.deepcopy(self.deformable_from_ref_low_res) if self.deformable_from_ref_low_res else None
        )
    
    def clear(self):
        """Clear all data references."""
        self.image = None
        self.image_low_res = None
        self.segmentation = None
        self.segmentation_mesh = None
        self.mask = None
        self.mask_low_res = None
        self.mask_high_res = None
        self.additional_meshes = None
        self.affine_matrix = None
        self.resliced_image = None
        self.resliced_segmentation_mesh = None
        self.resliced_meshes = None
        self.warp_image = None
        self.affine_from_prev = None
        self.affine_from_ref = None
        self.deformable_from_ref = None
        self.deformable_from_ref_low_res = None
