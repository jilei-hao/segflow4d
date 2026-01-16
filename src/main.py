import os
from propagation.propagation_pipeline import PropagationPipeline
from common.types.propagation_input import PropagationInputFactory
from utility.io.async_writer import async_writer
import logging

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def configure_logging(log_level='INFO', log_dir=''):
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'segflow4d.log')
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=log_level_int,
        format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s',
        handlers=handlers
    )

logger = logging.getLogger(__name__)

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="SegFlow4D Propagation Pipeline")
    parser.add_argument('--image4d', type=str, help='Path to the input 4d image')
    parser.add_argument('--output', type=str, help='Path to the output directory')
    parser.add_argument('--tp-ref', type=int, help='Reference time point')
    parser.add_argument('--tp-targets', type=int, nargs='+', help='Target time points')
    parser.add_argument('--seg-ref', type=str, help='Path to the segmentation reference image')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--log-dir', type=str, default='', help='Directory to store log files. Prints to console if not set.')
    parser.add_argument('--add-mesh', action='append', dest='additional_meshes', 
                        help='Add additional mesh: --add-mesh <name>:<path_to_mesh_file>')
    parser.add_argument('--lowres-factor', type=float, default=0.5, help='Low resolution resample factor')
    parser.add_argument('--dilation-radius', type=int, default=2, help='Dilation radius for segmentation')
    parser.add_argument('--registration-backend', type=str, default='fireants', help='Registration backend to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug-dir', type=str, default='', help='Directory to store debug outputs')

    parser.add_argument('--use-json-config', type=str, default='', help='Path to JSON configuration file')
    
    args = parser.parse_args()

    if args.additional_meshes:
        additional_meshes_dict = {}
        for item in args.additional_meshes:
            name, path = item.split(':', 1)
            additional_meshes_dict[name] = path
        args.additional_meshes = additional_meshes_dict

    return args

def load_json_config(config_path):
    """Load and parse JSON configuration file"""
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    args = parse_arguments()
    
    
    input_factory = PropagationInputFactory()

    if args.use_json_config:
        config = load_json_config(args.use_json_config)

        configure_logging(config.get('log_level', args.log_level), config.get('log_dir', args.log_dir))

        input_factory.set_image_4d_from_disk(config.get('image4d'))
        
        # Set global options
        input_factory.set_options(
            lowres_factor=config.get('lowres_factor', 0.5),
            registration_backend=config.get('registration_backend', 'fireants'),
            dilation_radius=config.get('dilation_radius', 2),
            write_result_to_disk=True,
            output_directory=config.get('output'),
            debug=config.get('debug', False),
            debug_output_directory=config.get('debug_dir', '')
        )
        
        # Parse and add multiple tp_input_groups
        tp_input_groups = config.get('tp_input_groups', [])
        
        if not tp_input_groups:
            logger.error("No tp_input_groups found in JSON configuration")
            return
        
        for group in tp_input_groups:
            tp_ref = group.get('tp_ref')
            tp_targets = group.get('tp_targets', [])
            seg_ref = group.get('seg_ref')
            additional_meshes = group.get('additional_meshes', {})
            
            logger.info(f"Adding tp_input_group with tp_ref={tp_ref}, tp_targets={tp_targets}")
            
            input_factory.add_tp_input_group_from_disk(
                tp_ref=tp_ref,
                tp_target=tp_targets,
                seg_ref_path=seg_ref,
                additional_meshes_ref=additional_meshes
            )
    else:
        configure_logging(args.log_level, args.log_dir)

        input_factory.set_image_4d_from_disk(args.image4d)
        # Use command-line arguments
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

if __name__ == "__main__":
    main()



