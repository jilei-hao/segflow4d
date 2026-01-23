from dataclasses import dataclass
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from common.types.tp_image_group import TPImageGroup
from common.types.tp_partition_input import TPPartitionInput
from common.types.propagation_options import PropagationOptions
from SimpleITK import Image, ReadImage
from vtkmodules.vtkCommonDataModel import vtkPolyData
from typing import Optional
from utility.mesh_helper.mesh_helper import read_polydata
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class PropagationInput:
    '''
    A class representing the input required for running the propagation pipeline.
    '''
    image_4d: Optional[ImageWrapper]
    tp_input_groups: list[TPPartitionInput]
    options: PropagationOptions

class PropagationInputFactory:
    def __init__(self):
        pass

    def set_image_4d(self, image_4d: Image) -> 'PropagationInputFactory':
        self._image_4d = ImageWrapper(image_4d)
        return self
    
    def set_image_4d_from_disk(self, file_path: str) -> 'PropagationInputFactory':
        logger.info(f"Loading 4D image from disk: {file_path}")
        image = ImageWrapper(ReadImage(file_path))
        self._image_4d = image
        return self
    
    def add_tp_input_group(self, tp_ref, tp_target, seg_ref: Image, additional_meshes_ref: Optional[dict[str, vtkPolyData]]) -> 'PropagationInputFactory':
        if not hasattr(self, '_tp_input_groups'):
            self._tp_input_groups = []

        # parse additional meshes
        _additional_meshes_ref = None
        if additional_meshes_ref is not None:
            _additional_meshes_ref = {k: MeshWrapper(v) for k, v in additional_meshes_ref.items()}

        tp_input_group = TPPartitionInput(
            seg_ref=ImageWrapper(seg_ref),
            additional_meshes_ref=_additional_meshes_ref,
            tp_ref=tp_ref,
            tp_target=tp_target
        )
        self._tp_input_groups.append(tp_input_group)
        return self
    
    
    def add_tp_input_group_from_disk(self, tp_ref, tp_target, seg_ref_path: str, additional_meshes_ref: dict[str, str]) -> 'PropagationInputFactory':
        logger.info(f"Adding TP input group from disk with seg_ref: {seg_ref_path}")
        if not hasattr(self, '_tp_input_groups'):
            self._tp_input_groups = []

        # parse additional meshes
        _additional_meshes_ref = None
        if additional_meshes_ref is not None:
            _additional_meshes_ref = {k: MeshWrapper(read_polydata(v)) for k, v in additional_meshes_ref.items()}

        tp_input_group = TPPartitionInput(
            seg_ref=ImageWrapper(ReadImage(seg_ref_path)),
            additional_meshes_ref=_additional_meshes_ref,
            tp_ref=tp_ref,
            tp_target=tp_target
        )
        self._tp_input_groups.append(tp_input_group)
        return self
    

    def set_options(self, lowres_factor: float, registration_backend: str, dilation_radius: int,
                    write_result_to_disk: bool = False, output_directory: str = "",
                    debug: bool = False, debug_output_directory: str = "", **kwargs) -> 'PropagationInputFactory':
        '''
        Sets the propagation options. *kwargs are reserved for configuring different registration backends.
        '''

        self._options = PropagationOptions(
            lowres_resample_factor=lowres_factor,
            dilation_radius=dilation_radius,
            registration_backend=registration_backend,
            write_result_to_disk=write_result_to_disk,
            output_directory=output_directory,
            debug=debug,
            debug_output_directory=debug_output_directory,
            registration_backend_options=kwargs
        )
        return self
    

    def build(self) -> PropagationInput:
        return PropagationInput(
            image_4d=self._image_4d,
            tp_input_groups=self._tp_input_groups,
            options=self._options
        )