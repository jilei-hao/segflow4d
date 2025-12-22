from .abstract_image_helper import AbstractImageHelper
from common.types.interpolation_type import InterpolationType, sitk_interpolation_from_type
import SimpleITK as sitk

class CPUImageHelper(AbstractImageHelper):
    def __init__(self):
        pass

    def binary_threshold(self, image: sitk.Image, lo: float, hi: float) -> sitk.Image:
        thresh_filter = sitk.BinaryThresholdImageFilter()
        thresh_filter.SetLowerThreshold(lo)
        thresh_filter.SetUpperThreshold(hi)
        thresh_filter.SetInsideValue(1)
        thresh_filter.SetOutsideValue(0)
        return thresh_filter.Execute(image)
    

    def resample(self, image: sitk.Image, resample_factor: float, interpolation: InterpolationType) -> sitk.Image:
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_size = [int(sz * resample_factor) for sz in original_size]
        new_spacing = [sp / resample_factor for sp in original_spacing]

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetOutputDirection(image.GetDirection())
        resample_filter.SetOutputOrigin(image.GetOrigin())
        resample_filter.SetInterpolator(sitk_interpolation_from_type(interpolation))

        return resample_filter.Execute(image)
    

    def binary_dilate(self, image: sitk.Image, radius: int) -> sitk.Image:
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(radius)
        dilate_filter.SetForegroundValue(1)
        return dilate_filter.Execute(image)