from .abstract_image_helper import AbstractImageHelper
from common.types.interpolation_type import InterpolationType, sitk_interpolation_from_type
from common.types.image_wrapper import ImageWrapper
import SimpleITK as sitk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.util.vtkConstants import VTK_UNSIGNED_CHAR, VTK_SHORT, VTK_UNSIGNED_SHORT, VTK_INT, VTK_UNSIGNED_INT, VTK_FLOAT, VTK_DOUBLE
from vtkmodules.util.numpy_support import numpy_to_vtk
import numpy as np

from logging import getLogger
logger = getLogger(__name__)


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
        image_data = image.get_data()
        if image_data is None:
            raise ValueError("Input image data is None.")

        original_size = image_data.GetSize()
        original_spacing = image_data.GetSpacing()
        new_size = [int(sz * resample_factor) for sz in original_size]
        new_spacing = [sp / resample_factor for sp in original_spacing]

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetOutputDirection(image_data.GetDirection())
        resample_filter.SetOutputOrigin(image_data.GetOrigin())
        resample_filter.SetInterpolator(sitk_interpolation_from_type(interpolation))

        return ImageWrapper(resample_filter.Execute(image_data))

    def binary_dilate(self, image: ImageWrapper, radius: int) -> ImageWrapper:
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(radius)
        dilate_filter.SetForegroundValue(1)
        image_data = image.get_data()
        if image_data is None:
            raise ValueError("Input image data is None.")
        return ImageWrapper(dilate_filter.Execute(image_data))

    def resample_to_reference(self, image: ImageWrapper, reference_image: ImageWrapper, interpolation: InterpolationType) -> ImageWrapper:
        resample_filter = sitk.ResampleImageFilter()
        data = reference_image.get_data()
        if data is None:
            raise ValueError("Reference image data is None.")

        image_data = image.get_data()
        if image_data is None:
            raise ValueError("Input image data is None.")

        resample_filter.SetSize(data.GetSize())
        resample_filter.SetOutputSpacing(data.GetSpacing())
        resample_filter.SetOutputDirection(data.GetDirection())
        resample_filter.SetOutputOrigin(data.GetOrigin())
        resample_filter.SetInterpolator(sitk_interpolation_from_type(interpolation))

        resampled = resample_filter.Execute(image_data)

        # Preserve original pixel type
        cast_filter = sitk.CastImageFilter()
        cast_filter.SetOutputPixelType(image_data.GetPixelID())
        return ImageWrapper(cast_filter.Execute(resampled))

    def extract_timepoint_image(self, image_4d: ImageWrapper, timepoint: int) -> ImageWrapper:
        extractor = sitk.ExtractImageFilter()
        image_data = image_4d.get_data()
        if image_data is None:
            raise ValueError("Input 4D image data is None.")
        size = list(image_data.GetSize())
        size[3] = 0  # Extract along the 4th dimension
        index = [0, 0, 0, timepoint - 1]  # timepoint is 1-based index
        extractor.SetSize(size)
        extractor.SetIndex(index)
        return ImageWrapper(extractor.Execute(image_data))

    def read_image(self, file_path: str) -> ImageWrapper:
        itk_image = sitk.ReadImage(file_path)
        return ImageWrapper(itk_image)

    def get_unique_labels(self, image: ImageWrapper) -> list[int]:
        if image is None:
            return []

        data = image.get_data()
        if data is None:
            return []

        array_data = sitk.GetArrayFromImage(data)
        unique_labels = list(set(array_data.flatten()))
        unique_labels.sort()
        return unique_labels

    def create_4d_image(self, image_list: list[ImageWrapper]) -> ImageWrapper:
        sitk_images = [img.get_data() for img in image_list]
        for i, img in enumerate(sitk_images):
            if img is None:
                raise ValueError(f"Image data at index {i} is None.")
        join_filter = sitk.JoinSeriesImageFilter()
        image_4d = join_filter.Execute(sitk_images)
        return ImageWrapper(image_4d)

    def convert_to_vtk_image(self, image: ImageWrapper) -> vtkImageData:
        image_data = image.get_data()
        if image_data is None:
            raise ValueError("Input image data is None.")

        array_data = sitk.GetArrayFromImage(image_data)  # shape: [D, H, W]
        spacing = image_data.GetSpacing()
        origin = image_data.GetOrigin()

        vtk_image = vtkImageData()
        
        depth, height, width = array_data.shape

        logger.debug(f"Converting image with dimensions (W x H x D): {width} x {height} x {depth}")
        
        vtk_image.SetDimensions(width, height, depth)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(origin)
        vtk_image.SetDirectionMatrix(image_data.GetDirection())

        # Transpose from [D, H, W] to [W, H, D] for VTK's x-fastest ordering
        # transposed_array = np.ascontiguousarray(array_data.transpose(2, 1, 0))
        flat_array = array_data.ravel(order='C')
        vtk_array = numpy_to_vtk(flat_array, deep=True)
        vtk_image.GetPointData().SetScalars(vtk_array)

        return vtk_image