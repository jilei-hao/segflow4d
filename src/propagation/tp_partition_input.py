from common.types.mesh_wrapper import MeshWrapper
from common.types.image_wrapper import ImageWrapper
from typing import Optional
from processing.segmentation_mesh_generator.multi_label_seg_mesh_generator import (
    MultiLabelSegMeshGenerator, MultiLabelSegMeshGeneratorOptions, MultiLabelSegMeshMethods
)
from utility.image_helper.image_helper_factory import create_image_helper
from logging import getLogger

logger = getLogger(__name__)

class TPPartitionInput:
    '''
    A class representing group of inputs for running propagation on one partition of all the timepoints.
    '''
    def __init__(self, 
        seg_ref: ImageWrapper,
        additional_meshes_ref: Optional[dict[str, MeshWrapper]],
        tp_ref: int,
        tp_target: list[int],
        seg_ref_mesh: Optional[MeshWrapper] = None
    ):
        self.seg_ref = seg_ref
        self.additional_meshes_ref = additional_meshes_ref
        self.tp_ref = tp_ref
        self.tp_target = tp_target

        if seg_ref_mesh is not None:
            self.seg_mesh_ref = seg_ref_mesh
        else:
            # generate mesh for reference segmentation
            ih = create_image_helper()
            labels = ih.get_unique_labels(self.seg_ref)
            labels.remove(0)  # remove background label

            ml_mesh_generator = MultiLabelSegMeshGenerator()
            options = MultiLabelSegMeshGeneratorOptions(
                label_list=labels,
                method=MultiLabelSegMeshMethods.DISCRETE_FLYING_EDGES
            )
            ml_mesh_generator.set_inputs(self.seg_ref, options)

            self.seg_mesh_ref = ml_mesh_generator.generate_mesh()

    
    def deepcopy(self) -> 'TPPartitionInput':
        # Deep copy the additional meshes dictionary and its contents
        if self.additional_meshes_ref:
            additional_meshes_copy = {
                key: mesh.deepcopy() for key, mesh in self.additional_meshes_ref.items()
            }
        else:
            additional_meshes_copy = None
        
        return TPPartitionInput(
            seg_ref=self.seg_ref.deepcopy(),
            additional_meshes_ref=additional_meshes_copy,
            tp_ref=self.tp_ref,
            tp_target=self.tp_target.copy(),  # list.copy() is also shallow, but ints are immutable so OK
            seg_ref_mesh=self.seg_mesh_ref.deepcopy()
        )