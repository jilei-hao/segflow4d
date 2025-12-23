from abc import ABC, abstractmethod
import SimpleITK as sitk
from common.types.interpolation_type import InterpolationType

class AbstractImageHelper(ABC):
    @abstractmethod
    def binary_threshold(self, image: sitk.Image, lo: float, hi: float) -> sitk.Image:
        pass

    @abstractmethod
    def resample(self, image: sitk.Image, resample_factor: float, interpolation: InterpolationType) -> sitk.Image:
        pass

    @abstractmethod
    def binary_dilate(self, image: sitk.Image, radius: int) -> sitk.Image:
        pass

    @abstractmethod
    def resample_to_reference(self, image: sitk.Image, reference_image: sitk.Image, interpolation: InterpolationType) -> sitk.Image:
        pass