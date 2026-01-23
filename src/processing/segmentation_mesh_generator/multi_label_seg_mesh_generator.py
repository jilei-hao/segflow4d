from processing.segmentation_mesh_generator.abstract_segmentation_mesh_generator import AbstractSegmentationMeshGenerator
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from dataclasses import dataclass
from utility.image_helper.image_helper_factory import create_image_helper
import numpy as np
import vtk
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes, vtkDiscreteFlyingEdges3D
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.util.numpy_support import numpy_to_vtk
from logging import getLogger

logger = getLogger(__name__)


class MultiLabelSegMeshMethods:
    DISCRETE_MARCHING_CUBES = "discrete_marching_cubes"
    DISCRETE_FLYING_EDGES = "discrete_flying_edges"


@dataclass
class MultiLabelSegMeshGeneratorOptions:
    label_list: list[int]
    method: str = MultiLabelSegMeshMethods.DISCRETE_MARCHING_CUBES


class MultiLabelSegMeshGenerator(AbstractSegmentationMeshGenerator):
    def set_inputs(self, image: ImageWrapper, options: MultiLabelSegMeshGeneratorOptions):
        """
        Set the input image and options for multi-label segmentation mesh generation.

        Args:
            image: The input image to be segmented.
            options: An instance of MultiLabelSegMeshGeneratorOptions containing options for segmentation.
        """
        self.image = image
        self.options = options

    def generate_mesh(self) -> MeshWrapper:
        """
        Generate the multi-label segmentation mesh based on the input image and options.

        Returns:
            The generated segmentation mesh.
        """
        logger.info("Starting multi-label segmentation mesh generation")

        # Get image data and spatial information
        image_data = self.image.get_data()  # numpy array [D, H, W]
        if image_data is None:
            raise ValueError("Image data is None")
        # Convert to numpy array if needed
        origin = self.image.get_origin()
        spacing = self.image.get_spacing()
        
        # Validate origin and spacing
        if origin is None:
            raise ValueError("Image origin is None")
        if spacing is None:
            raise ValueError("Image spacing is None")
        
        # Convert numpy array to VTK image data
        logger.info("Converting sitk image to VTK image data")
        image_helper = create_image_helper()
        vtk_image = image_helper.convert_to_vtk_image(self.image)
        logger.info("VTK image data conversion completed")
        
        # Choose the extraction method
        logger.info(f"Using method: {self.options.method} for mesh generation")
        if self.options.method == MultiLabelSegMeshMethods.DISCRETE_FLYING_EDGES:
            extractor = vtkDiscreteFlyingEdges3D()
        else:
            extractor = vtkDiscreteMarchingCubes()
        
        extractor.SetInputData(vtk_image)
        
        # Set the label values to extract
        for i, label in enumerate(self.options.label_list):
            extractor.SetValue(i, label)
        
        extractor.Update()
        logger.info("Mesh extraction completed")
        
        # Get the output polydata
        output_polydata = extractor.GetOutput()
        
        # Add label information as cell data
        self._add_label_cell_data(output_polydata)
        logger.info("Label cell data added to mesh")
        
        return MeshWrapper(output_polydata)

    def _numpy_to_vtk_image(
        self, 
        data: np.ndarray, 
        origin: tuple, 
        spacing: tuple
    ) -> vtk.vtkImageData:
        """
        Convert a numpy array to VTK image data with proper spatial information.

        Args:
            data: numpy array [D, H, W]
            origin: image origin (x, y, z)
            spacing: image spacing (x, y, z)

        Returns:
            vtkImageData with proper spatial information
        """
        vtk_image = vtk.vtkImageData()
        
        # VTK expects dimensions as (W, H, D)
        depth, height, width = data.shape
        vtk_image.SetDimensions(width, height, depth)
        vtk_image.SetOrigin(origin)
        vtk_image.SetSpacing(spacing)
        
        # Flatten and convert to VTK array
        # Note: need to transpose for VTK's expected order
        flat_data = np.ascontiguousarray(data.ravel(order='F'))
        vtk_array = numpy_to_vtk(flat_data, deep=True)
        vtk_array.SetName("Labels")
        
        vtk_image.GetPointData().SetScalars(vtk_array)
        
        return vtk_image

    def _add_label_cell_data(self, polydata: vtkPolyData):
        """
        Add label information as cell data based on scalar values.

        Args:
            polydata: The polydata to add label data to
        """
        # The discrete marching cubes already assigns scalar values
        # corresponding to the label values, so we just rename the array
        scalars = polydata.GetPointData().GetScalars()
        if scalars:
            scalars.SetName("Label")
            
            # Also add as cell data by averaging point data
            point_to_cell = vtk.vtkPointDataToCellData()
            point_to_cell.SetInputData(polydata)
            point_to_cell.Update()
            
            cell_scalars = point_to_cell.GetOutput().GetCellData().GetScalars()
            if cell_scalars:
                cell_scalars.SetName("Label")
                polydata.GetCellData().SetScalars(cell_scalars)