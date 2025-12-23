from dataclasses import dataclass
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from common.types.tp_image_group import TPImageGroup
from common.types.tp_partition_input import TPPartitionInputGroup
from common.types.propagation_options import PropagationOptions
from SimpleITK import Image, ReadImage
from vtkmodules.vtkCommonDataModel import vtkPolyData
from typing import Optional
from utility.mesh_helper import read_polydata

@dataclass
class PropagationInput:
    '''
    A class representing the input required for running the propagation pipeline.
    '''
    image_4d: ImageWrapper
    tp_input_groups: list[TPPartitionInputGroup]
    options: PropagationOptions

class PropagationInputFactory:
    def __init__(self):
        pass

    def set_image_4d(self, image_4d: Image) -> 'PropagationInputFactory':
        self._image_4d = ImageWrapper(image_4d)
        return self
    
    def set_image_4d_from_disk(self, file_path: str) -> 'PropagationInputFactory':
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

        tp_input_group = TPPartitionInputGroup(
            seg_ref=ImageWrapper(seg_ref),
            seg_mesh_ref=None,
            additional_meshes_ref=_additional_meshes_ref,
            tp_ref=tp_ref,
            tp_target=tp_target
        )
        self._tp_input_groups.append(tp_input_group)
        return self
    
    
    def add_tp_input_group_from_disk(self, tp_ref, tp_target, seg_ref_path: str, seg_mesh_ref: str, additional_meshes_ref: dict[str, str]) -> 'PropagationInputFactory':
        if not hasattr(self, '_tp_input_groups'):
            self._tp_input_groups = []

        # parse additional meshes
        _additional_meshes_ref = None
        if additional_meshes_ref is not None:
            _additional_meshes_ref = {k: MeshWrapper(read_polydata(v)) for k, v in additional_meshes_ref.items()}

        tp_input_group = TPPartitionInputGroup(
            seg_ref=ImageWrapper(ReadImage(seg_ref_path)),
            seg_mesh_ref=MeshWrapper(read_polydata(seg_mesh_ref)) if seg_mesh_ref is not None else None,
            additional_meshes_ref=_additional_meshes_ref,
            tp_ref=tp_ref,
            tp_target=tp_target
        )
        self._tp_input_groups.append(tp_input_group)
        return self
    

    def set_options(self, lowres_factor: float, dilation_radius: int, **kwargs) -> 'PropagationInputFactory':
        '''
        Sets the propagation options. *kwargs are reserved for configuring different registration backends.
        '''

        self._options = PropagationOptions(
            lowres_resample_factor=lowres_factor,
            dilation_radius=dilation_radius,
            registration_backend_options=kwargs
        )
        return self
    

    def build(self) -> PropagationInput:
        return PropagationInput(
            image_4d=self._image_4d,
            tp_input_groups=self._tp_input_groups,
            options=self._options
        )