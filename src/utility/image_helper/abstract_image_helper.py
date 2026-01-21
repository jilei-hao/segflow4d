from abc import ABC, abstractmethod
from common.types.interpolation_type import InterpolationType
from common.types.image_wrapper import ImageWrapper

class AbstractImageHelper(ABC):
    @abstractmethod
    def binary_threshold(self, image: ImageWrapper, lo: float, hi: float) -> ImageWrapper:
        pass

    @abstractmethod
    def resample(self, image: ImageWrapper, resample_factor: float, interpolation: InterpolationType) -> ImageWrapper:
        pass

    @abstractmethod
    def binary_dilate(self, image: ImageWrapper, radius: int) -> ImageWrapper:
        pass

    @abstractmethod
    def resample_to_reference(self, image: ImageWrapper, reference_image: ImageWrapper, interpolation: InterpolationType) -> ImageWrapper:
        pass

    @abstractmethod
    def extract_timepoint_image(self, image_4d: ImageWrapper, timepoint: int) -> ImageWrapper:
        pass

    @abstractmethod
    def read_image(self, file_path: str) -> ImageWrapper:
        pass

    @abstractmethod
    def get_unique_labels(self, image: ImageWrapper) -> list[int]:
        pass


    @abstractmethod
    def create_4d_image(self, image_list: list[ImageWrapper]) -> ImageWrapper:
        pass