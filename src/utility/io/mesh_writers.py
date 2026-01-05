from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter

def vtk_polydata_writer(polydata: vtkPolyData, file_path: str):
    """Writes a VTK PolyData object to a file."""
    if file_path.endswith('.vtp'):
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(polydata)
        writer.Write()
        return
    
    writer = vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()
