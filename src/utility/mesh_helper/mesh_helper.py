import vtk

def read_polydata(file_path: str) -> vtk.vtkPolyData:
    """Reads a VTK PolyData file and returns the PolyData object."""
    if file_path.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file_path)
        reader.Update()
        return reader.GetOutput()
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()


def write_polydata(polydata: vtk.vtkPolyData, file_path: str):
    """Writes a VTK PolyData object to a file."""
    if file_path.endswith('.vtp'):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(polydata)
        writer.Write()
        return

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()