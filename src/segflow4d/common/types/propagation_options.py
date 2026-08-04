from dataclasses import dataclass, field
from typing import Any
from segflow4d.common.types.abstract_registration_options import AbstractRegistrationOptions

@dataclass
class PropagationOptions:
    '''
    A class representing options for propagation.
    '''
    lowres_scale_factor: float
    dilation_radius: int
    registration_backend: str
    registration_backend_options: AbstractRegistrationOptions
    write_result_to_disk: bool = False
    output_directory: str = ""
    debug: bool = False
    debug_output_directory: str = ""
    minimum_required_vram_gb: int = 10
    propagation_strategy_combo: str = "sequential_star"
    # When > 0, crop the high-res images / masks / segmentation to the union
    # bbox of the propagated low-res masks (padded by this many voxels) before
    # running the high-res registration stage, then uncrop the resliced
    # segmentation back to the reference frame. 0 disables ROI cropping.
    roi_crop_padding_voxels: int = 0
    # Whether to hand the propagated masks to the high-res registration stage.
    # The backend applies a mask as a hard intensity multiply on BOTH images
    # (fireants_registration_handler.py:410-417), and fixed and moving get
    # DIFFERENT masks -- the target's propagated mask vs the reference's. The
    # two masked images therefore have different dilated-blob silhouettes, and
    # an MSE objective is dominated by aligning those silhouettes rather than
    # the anatomy inside them. When ROI cropping is on, the crop already
    # restricts the registration to the root, so the mask is largely redundant.
    # Masks are still used for the ROI crop bbox regardless of this setting.
    mask_high_res_registration: bool = True

    def __post_init__(self):
        pass