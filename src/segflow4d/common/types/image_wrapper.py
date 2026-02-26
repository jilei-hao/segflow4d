from SimpleITK import Image
import SimpleITK as sitk
import numpy as np
from segflow4d.common.types.propagation_data_object import PropagationDataObject


class ImageWrapper(PropagationDataObject):
    def __init__(self, image: Image | None):
        super().__init__()
        self._data = image

    def get_data(self) -> Image | None:
        return self._data
    
    def set_data(self, data: Image | None):
        self._data = data

    def get_origin(self) -> tuple | None:
        if self._data is None:
            return None
        return self._data.GetOrigin()
    
    def get_spacing(self) -> tuple | None:
        if self._data is None:
            return None
        return self._data.GetSpacing()
    
    def get_direction(self) -> tuple | None:
        if self._data is None:
            return None
        return self._data.GetDirection()
    
    def get_dimensions(self) -> tuple | None:
        if self._data is None:
            return None
        return self._data.GetSize()


    def deepcopy(self) -> 'ImageWrapper':
        """Create a deep copy of ImageWrapper with independent pixel data."""
        if self._data is None:
            return ImageWrapper(None)
        
        # Extract pixel data as numpy array
        array_data = sitk.GetArrayFromImage(self._data)
        
        # Create a deep copy of the numpy array
        array_copy = np.copy(array_data)
        
        # Create new ITK image from the copied array
        image_copy = sitk.GetImageFromArray(array_copy)
        
        # Copy metadata from original image
        image_copy.SetSpacing(self._data.GetSpacing())
        image_copy.SetOrigin(self._data.GetOrigin())
        image_copy.SetDirection(self._data.GetDirection())
        
        return ImageWrapper(image_copy)

