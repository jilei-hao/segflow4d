from abc import ABC, abstractmethod

from common.types.propagation_options import PropagationOptions

class AbstractRegistrationHandler(ABC):
    @abstractmethod
    def run_affine(self, img_fixed, img_moving, options: PropagationOptions):
        pass

    @abstractmethod
    def run_deformable(self, img_fixed, img_moving, options: PropagationOptions):
        pass

    @abstractmethod
    def run_reslice_segmentation(self, img_to_reslice, img_reference,  options: PropagationOptions):
        pass


    @abstractmethod
    def run_reslice_mesh(self, mesh_to_reslice, img_reference, options: PropagationOptions):
        pass


    @abstractmethod
    def run_registration_and_reslice(self, img_fixed, img_moving, img_to_reslice, mesh_to_reslice, options: PropagationOptions, mask_fixed = None, mask_moving = None) -> dict:
        pass


    @abstractmethod
    def get_device_type(self) -> str:
        pass