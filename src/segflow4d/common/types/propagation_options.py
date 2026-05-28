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

    def __post_init__(self):
        pass