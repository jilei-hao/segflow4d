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
        
        return self  # Return self to allow method chaining and assignment


    def deepcopy(self) -> 'MeshWrapper':
        mesh_copy = vtkPolyData()
        mesh_copy.DeepCopy(self._data)
        return MeshWrapper(mesh_copy)


    def __getstate__(self):
        """Prepare object for pickling by converting VTK data to serializable format."""
        from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
        import io
        import base64
        
        # Write VTK data to string
        writer = vtkXMLPolyDataWriter()
        writer.SetInputData(self._data)
        writer.SetWriteToOutputString(True)
        writer.Write()
        
        # Get the XML string and encode to bytes
        xml_string = writer.GetOutputString()
        
        return {'mesh_data': xml_string}


    def __setstate__(self, state):
        """Restore object from pickled state by reconstructing VTK data."""
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
        
        # Read VTK data from string
        reader = vtkXMLPolyDataReader()
        reader.SetReadFromInputString(True)
        reader.SetInputString(state['mesh_data'])
        reader.Update()
        
        self._data = reader.GetOutput()