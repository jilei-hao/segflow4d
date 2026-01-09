from propagation.propagation_pipeline import PropagationPipeline
from common.types.propagation_input import PropagationInputFactory
from utility.io.async_writer import async_writer
import logging

def configure_logging(log_level='INFO'):
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level_int,
        format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="SegFlow4D Propagation Pipeline")
    parser.add_argument('--image4d', type=str, required=True, help='Path to the input 4d image')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--tp-ref', type=int, required=True, help='Reference time point')
    parser.add_argument('--tp-targets', type=int, nargs='+', required=True, help='Target time points')
    parser.add_argument('--seg-ref', type=str, required=True, help='Path to the segmentation reference image')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--add-mesh', action='append', dest='additional_meshes', 
                        help='Add additional mesh: --add-mesh <name>:<path_to_mesh_file>')
    parser.add_argument('--lowres-factor', type=float, default=0.5, help='Low resolution resample factor')
    parser.add_argument('--dilation-radius', type=int, default=2, help='Dilation radius for segmentation')
    parser.add_argument('--registration-backend', type=str, default='fireants', help='Registration backend to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug-dir', type=str, default='', help='Directory to store debug outputs')
    args = parser.parse_args()

    if args.additional_meshes:
        additional_meshes_dict = {}
        for item in args.additional_meshes:
            name, path = item.split(':', 1)
            additional_meshes_dict[name] = path
        args.additional_meshes = additional_meshes_dict

    return args

def main():
    args = parse_arguments()
    configure_logging(args.log_level)

    logger = logging.getLogger("segflow4d")

    input_factory = PropagationInputFactory()
    input_factory.set_image_4d_from_disk(args.image4d)
    input_factory.add_tp_input_group_from_disk(
        tp_ref=args.tp_ref,
        tp_target=args.tp_targets,
        seg_ref_path=args.seg_ref,
        additional_meshes_ref=args.additional_meshes
    )
    input_factory.set_options(
        lowres_factor=args.lowres_factor,
        registration_backend=args.registration_backend,
        dilation_radius=args.dilation_radius,
        write_result_to_disk=True,
        output_directory=args.output,
        debug=args.debug,
        debug_output_directory=args.debug_dir
    )

    propagation_input = input_factory.build()
    pipeline = PropagationPipeline(propagation_input)

    logger.info("Starting SegFlow4D Propagation Pipeline")
    pipeline.run()

    
    async_writer.shutdown(wait=True)
    logger.info("SegFlow4D Propagation Pipeline completed")



