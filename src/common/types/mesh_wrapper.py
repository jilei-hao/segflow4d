from vtkmodules.vtkCommonDataModel import vtkPolyData
from .propagation_data_object import PropagationDataObject

class MeshWrapper(PropagationDataObject):
    def __init__(self, mesh: vtkPolyData):
        super().__init__()
        self._data = mesh


    def get_data(self) -> vtkPolyData:
        return self._data
    

    def set_data(self, data: vtkPolyData):
        self._data = data