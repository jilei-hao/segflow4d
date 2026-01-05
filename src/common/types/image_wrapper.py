from SimpleITK import Image
from .propagation_data_object import PropagationDataObject

class ImageWrapper(PropagationDataObject):
    def __init__(self, image: Image):
        super().__init__()
        self._data = image


    def get_data(self) -> Image:
        return self._data
    

    def set_data(self, data: Image):
        self._data = data

    