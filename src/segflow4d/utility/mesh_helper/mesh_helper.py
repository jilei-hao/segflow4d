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


def change_coordinate_system(polydata: vtk.vtkPolyData, from_system: str, to_system: str) -> vtk.vtkPolyData:
    """Changes the coordinate system of the given PolyData."""
    transform = vtk.vtkTransform()

    if from_system == "LPS" and to_system == "RAS":
        transform.Scale(-1, -1, 1)
    elif from_system == "RAS" and to_system == "LPS":
        transform.Scale(-1, -1, 1)
    elif from_system == "RAI" and to_system == "LPS":
        transform.Scale(-1, 1, 1)
    elif from_system == "LPS" and to_system == "RAI":
        transform.Scale(-1, 1, 1)
    else:
        raise ValueError(f"Unsupported coordinate system transformation from {from_system} to {to_system}")

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()