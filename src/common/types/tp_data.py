from typing import Optional
import copy

from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper

class TPData:
    '''
    A class representing the data for one timepoint.
    '''
    def __init__(
        self,
        image: ImageWrapper,
        image_low_res: Optional[ImageWrapper] = None,
        segmentation: Optional[ImageWrapper] = None,
        segmentation_mesh: Optional[MeshWrapper] = None,
        mask_low_res: Optional[ImageWrapper] = None,
        mask_high_res: Optional[ImageWrapper] = None,
        additional_meshes: Optional[dict[str, MeshWrapper]] = None,
        affine_from_prev: Optional[object] = None,
        affine_from_ref: Optional[object] = None,
        deformable_from_ref: Optional[object] = None,
        deformable_from_ref_low_res: Optional[object] = None
    ):
        # result objects
        self.image = image
        self.image_low_res = image_low_res
        self.segmentation = segmentation
        self.segmentation_mesh = segmentation_mesh
        self.mask_low_res = mask_low_res
        self.mask_high_res = mask_high_res
        self.additional_meshes = additional_meshes

        # transformation objects
        self.affine_from_prev = affine_from_prev
        self.affine_from_ref = affine_from_ref
        self.deformable_from_ref = deformable_from_ref
        self.deformable_from_ref_low_res = deformable_from_ref_low_res

    def deepcopy(self) -> 'TPData':
        """Create a deep copy of TPData with copied ImageWrapper and MeshWrapper objects."""
        return TPData(
            image=self.image.deepcopy(),
            image_low_res=self.image_low_res.deepcopy() if self.image_low_res else None,
            segmentation=self.segmentation.deepcopy() if self.segmentation else None,
            segmentation_mesh=self.segmentation_mesh.deepcopy() if self.segmentation_mesh else None,
            mask_low_res=self.mask_low_res.deepcopy() if self.mask_low_res else None,
            mask_high_res=self.mask_high_res.deepcopy() if self.mask_high_res else None,
            additional_meshes={k: v.deepcopy() for k, v in self.additional_meshes.items()} if self.additional_meshes else None,
            affine_from_prev=copy.deepcopy(self.affine_from_prev) if self.affine_from_prev else None,
            affine_from_ref=copy.deepcopy(self.affine_from_ref) if self.affine_from_ref else None,
            deformable_from_ref=copy.deepcopy(self.deformable_from_ref) if self.deformable_from_ref else None,
            deformable_from_ref_low_res=copy.deepcopy(self.deformable_from_ref_low_res) if self.deformable_from_ref_low_res else None
        )
