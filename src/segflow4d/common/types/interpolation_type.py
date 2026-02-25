from enum import Enum
import SimpleITK as sitk

class InterpolationType(Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    BSPLINE = "bspline" 
    GAUSSIAN = "gaussian"


def sitk_interpolation_from_type(interp_type: InterpolationType) -> int:
    if interp_type == InterpolationType.NEAREST:
        return sitk.sitkNearestNeighbor
    elif interp_type == InterpolationType.LINEAR:
        return sitk.sitkLinear
    elif interp_type == InterpolationType.BSPLINE:
        return sitk.sitkBSpline
    elif interp_type == InterpolationType.GAUSSIAN:
        return sitk.sitkGaussian
    else:
        raise ValueError(f"Unsupported interpolation type: {interp_type}")