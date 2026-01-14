from .abstract_image_helper import AbstractImageHelper
from common.types.interpolation_type import InterpolationType, sitk_interpolation_from_type
from common.types.image_wrapper import ImageWrapper
import SimpleITK as sitk


class CPUImageHelper(AbstractImageHelper):
    def __init__(self):
        pass

    def binary_threshold(self, image: ImageWrapper, lo: float, hi: float) -> ImageWrapper:
        thresh_filter = sitk.BinaryThresholdImageFilter()
        thresh_filter.SetLowerThreshold(lo)
        thresh_filter.SetUpperThreshold(hi)
        thresh_filter.SetInsideValue(1)
        thresh_filter.SetOutsideValue(0)
        return ImageWrapper(thresh_filter.Execute(image.get_data()))
    

    def resample(self, image: ImageWrapper, resample_factor: float, interpolation: InterpolationType) -> ImageWrapper:
        
        original_size = image.get_data().GetSize()
        original_spacing = image.get_data().GetSpacing()
        new_size = [int(sz * resample_factor) for sz in original_size]
        new_spacing = [sp / resample_factor for sp in original_spacing]

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetOutputDirection(image.get_data().GetDirection())
        resample_filter.SetOutputOrigin(image.get_data().GetOrigin())
        resample_filter.SetInterpolator(sitk_interpolation_from_type(interpolation))

        return ImageWrapper(resample_filter.Execute(image.get_data()))
    

    def binary_dilate(self, image: ImageWrapper, radius: int) -> ImageWrapper:
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(radius)
        dilate_filter.SetForegroundValue(1)
        return ImageWrapper(dilate_filter.Execute(image.get_data()))
    

    def resample_to_reference(self, image: ImageWrapper, reference_image: ImageWrapper, interpolation: InterpolationType) -> ImageWrapper:
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetSize(reference_image.get_data().GetSize())
        resample_filter.SetOutputSpacing(reference_image.get_data().GetSpacing())
        resample_filter.SetOutputDirection(reference_image.get_data().GetDirection())
        resample_filter.SetOutputOrigin(reference_image.get_data().GetOrigin())
        resample_filter.SetInterpolator(sitk_interpolation_from_type(interpolation))

        resampled = resample_filter.Execute(image.get_data())
        
        # Preserve original pixel type
        cast_filter = sitk.CastImageFilter()
        cast_filter.SetOutputPixelType(image.get_data().GetPixelID())
        return ImageWrapper(cast_filter.Execute(resampled))
    
    def extract_timepoint_image(self, image_4d: ImageWrapper, timepoint: int) -> ImageWrapper:
        extractor = sitk.ExtractImageFilter()
        size = list(image_4d.get_data().GetSize())
        size[3] = 0  # Extract along the 4th dimension
        index = [0, 0, 0, timepoint - 1]    # timepoint is 1-based index
        extractor.SetSize(size)
        extractor.SetIndex(index)
        return ImageWrapper(extractor.Execute(image_4d.get_data()))
    

    def read_image(self, file_path: str) -> ImageWrapper:
        itk_image = sitk.ReadImage(file_path)
        return ImageWrapper(itk_image)
    

    def get_unique_labels(self, image: ImageWrapper) -> list[int]:
        array_data = sitk.GetArrayFromImage(image.get_data())
        unique_labels = list(set(array_data.flatten()))
        unique_labels.sort()
        return unique_labels