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
    # Upper bound on registration jobs run at once on each GPU (process-pool
    # dispatcher). 1 keeps the original one-at-a-time behaviour. A job spends much of
    # its time off the GPU (IPC, host-side prep and post-processing), and another
    # job's GPU work can fill that time.
    #
    # Above 1, packing is memory-aware, so the same setting is safe for small and
    # large volumes. Each worker reports its job's VRAM peak (allocator plus CUDA
    # context). A further job joins only if the measured, or extrapolated, peaks of
    # the jobs on the device fit the free VRAM seen when it was idle, less 512 MB.
    # The first job of a new kind runs alone to be measured. A job that still runs out
    # of memory next to others is retried alone. Measured on an 8 GB card (bavcta005):
    #   ROI crop 16: 172 s at 1, 135 s at 2.
    #   ROI crop 64: 248 s at 1, 216 s at 2.
    #   ROI crop 64 with ~5 GB free: completes at 2, by running big jobs alone; a
    #   fixed two-at-a-time dispatcher ran out of memory.
    gpu_jobs_per_device: int = 1

    def __post_init__(self):
        pass