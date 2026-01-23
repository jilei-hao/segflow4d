from processing.segmentation_mesh_generator.multi_label_seg_mesh_generator import MultiLabelSegMeshGenerator, MultiLabelSegMeshGeneratorOptions, MultiLabelSegMeshMethods
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from utility.image_helper.image_helper_factory import create_image_helper
from utility.mesh_helper.mesh_helper import write_polydata
from logging import getLogger, basicConfig

basicConfig(level="INFO")
logger = getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-label Segmentation Mesh Generator CLI")
    parser.add_argument("--input-image", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--output-mesh", type=str, required=True, help="Path to save the output mesh file")
    parser.add_argument("--labels", type=int, nargs='+', required=True, help="List of label values to generate meshes for")
    parser.add_argument("--method", type=str, choices=[MultiLabelSegMeshMethods.DISCRETE_MARCHING_CUBES, MultiLabelSegMeshMethods.DISCRETE_FLYING_EDGES], default=MultiLabelSegMeshMethods.DISCRETE_FLYING_EDGES, help="Mesh generation method")
    args = parser.parse_args() 

    # Load input image
    ih = create_image_helper()
    image = ih.read_image(args.input_image)

    logger.info("Input image loaded successfully")
    
    # Set up mesh generator
    mesh_generator = MultiLabelSegMeshGenerator()
    options = MultiLabelSegMeshGeneratorOptions(label_list=args.labels, method=args.method)
    mesh_generator.set_inputs(image, options)
    mesh = mesh_generator.generate_mesh()
    write_polydata(mesh.get_data(), args.output_mesh)

    logger.info("Mesh generation completed and saved to output")

if __name__ == "__main__":
    main()