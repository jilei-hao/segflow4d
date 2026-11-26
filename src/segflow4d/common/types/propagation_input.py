from dataclasses import dataclass
import os
from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.mesh_wrapper import MeshWrapper
from segflow4d.common.types.tp_image_group import TPImageGroup
from segflow4d.propagation.tp_partition_input import TPPartitionInput
from segflow4d.common.types.propagation_options import PropagationOptions
from SimpleITK import Image, ReadImage
from vtkmodules.vtkCommonDataModel import vtkPolyData
from segflow4d.utility.mesh_helper.mesh_helper import read_polydata
from segflow4d.utility.mesh_helper.mesh_placement import check_mesh_placement
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class PropagationInput:
    '''
    A class representing the input required for running the propagation pipeline.
    '''
    image_4d: ImageWrapper | None
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
    
    def add_tp_input_group(self, tp_ref, tp_target, seg_ref: Image, additional_meshes_ref: dict[str, vtkPolyData] | None) -> 'PropagationInputFactory':
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
            _additional_meshes_ref = {}
            for k, v in additional_meshes_ref.items():
                poly = read_polydata(v)
                if poly.GetNumberOfPoints() == 0:
                    raise ValueError(
                        f"Additional mesh {k!r} read from {v} has no vertices. Check that the "
                        f"file holds VTK PolyData: a legacy .vtk file can also hold an "
                        f"unstructured grid, which reads as empty."
                    )
                _additional_meshes_ref[k] = MeshWrapper(poly)

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
                    debug: bool = False, debug_output_directory: str = "",
                    minimum_required_vram_gb: int = 10,
                    propagation_strategy_combo: str = "sequential_star",
                    roi_crop_padding_voxels: int = 0,
                    cyclic_time: bool = False,
                    mask_high_res_registration: bool = True,
                    gpu_jobs_per_device: int = 1,
                    **kwargs) -> 'PropagationInputFactory':
        '''
        Sets the propagation options. *kwargs are reserved for configuring different registration backends.
        '''

        self._options = PropagationOptions(
            lowres_scale_factor=lowres_factor,
            dilation_radius=dilation_radius,
            registration_backend=registration_backend,
            write_result_to_disk=write_result_to_disk,
            output_directory=output_directory,
            debug=debug,
            debug_output_directory=debug_output_directory,
            minimum_required_vram_gb=minimum_required_vram_gb,
            registration_backend_options=kwargs,
            propagation_strategy_combo=propagation_strategy_combo,
            roi_crop_padding_voxels=roi_crop_padding_voxels,
            cyclic_time=cyclic_time,
            mask_high_res_registration=mask_high_res_registration,
            gpu_jobs_per_device=gpu_jobs_per_device,
        )

        os.makedirs(output_directory, exist_ok=True)
        if debug:
            os.makedirs(debug_output_directory, exist_ok=True)
        return self
    

    def build(self) -> PropagationInput:
        # A mesh outside the image would be copied unchanged to every timepoint,
        # silently; most often it is a RAS mesh. Fail here instead (see mesh_placement).
        for group in getattr(self, '_tp_input_groups', []):
            for name, mesh in (group.additional_meshes_ref or {}).items():
                if mesh.get_data().GetNumberOfPoints() == 0:
                    # Nothing to place or warp. An in-memory caller (e.g. avrp-handler
                    # meshing a tiny label) may legitimately pass one, and earlier versions
                    # let it through, so warn rather than fail. An unreadable FILE raises
                    # in add_tp_input_group_from_disk instead.
                    logger.warning(f"Additional mesh {name!r} has no vertices; it is propagated empty.")
                    continue
                check_mesh_placement(name, mesh.get_vertices(), group.seg_ref.get_data())
        return PropagationInput(
            image_4d=self._image_4d,
            tp_input_groups=self._tp_input_groups,
            options=self._options
        )