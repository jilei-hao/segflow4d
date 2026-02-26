from enum import Enum
import SimpleITK as sitk

class InterpolationType(Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    BSPLINE = "bspline" 
    GAUSSIAN = "gaussian"


def sitk_interpolation_from_type(interp_type: InterpolationType) -> int:
    match interp_type:
        case InterpolationType.NEAREST:
            return sitk.sitkNearestNeighbor
        case InterpolationType.LINEAR:
            return sitk.sitkLinear
        case InterpolationType.BSPLINE:
            return sitk.sitkBSpline
        case InterpolationType.GAUSSIAN:
            return sitk.sitkGaussian
        case _:
            raise ValueError(f"Unsupported interpolation type: {interp_type}")