from vtkmodules.vtkCommonDataModel import vtkPolyData
from .propagation_data_object import PropagationDataObject
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

class MeshWrapper(PropagationDataObject):
    def __init__(self, mesh: vtkPolyData):
        super().__init__()
        self._data = mesh


    def get_data(self) -> vtkPolyData:
        return self._data
    

    def set_data(self, data: vtkPolyData):
        self._data = data


    def get_vertices(self) -> np.ndarray:
        points = self._data.GetPoints()
        # Direct conversion to NumPy array (no loop)
        vertices = vtk_to_numpy(points.GetData())
        return vertices.astype(np.float32)
    

    def update_vertices(self, new_vertices: np.ndarray):
        # Ensure array is contiguous and float64 (VTK default)
        new_vertices = np.ascontiguousarray(new_vertices, dtype=np.float64)
        
        # Convert NumPy array directly to VTK array
        vtk_array = numpy_to_vtk(new_vertices, deep=True)
        vtk_array.SetNumberOfComponents(3)
        
        # Replace the entire points array
        self._data.GetPoints().SetData(vtk_array)
        self._data.Modified()


    def deepcopy(self) -> 'MeshWrapper':
        mesh_copy = vtkPolyData()
        mesh_copy.DeepCopy(self._data)
        return MeshWrapper(mesh_copy)