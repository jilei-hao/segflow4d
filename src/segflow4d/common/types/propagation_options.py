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
    # Treat the timepoint axis as a closed loop, so a propagation chain may walk
    # off the end of the series and continue at timepoint 1 (and vice versa).
    #
    # For gated cardiac series the last frame is a neighbour of the first, and a
    # group whose membership straddles that wrap point (e.g. ref 8 owning
    # ... 19, 20, 1, 2) otherwise has no route to its wrapped members: the chain
    # is forced to jump straight from the nearest in-group frame, across frames
    # owned by another group. On bavcta025 that produced a single 11.9 mm step
    # (7 -> 2) where every real adjacent step is under 3.7 mm.
    #
    # Off by default: enabling it changes which registrations are performed, and
    # it is only meaningful for genuinely cyclic acquisitions.
    cyclic_time: bool = False

    def __post_init__(self):
        pass