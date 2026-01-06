from registration.abstract_registration_handler import AbstractRegistrationHandler

class GreedyRegistrationHandler(AbstractRegistrationHandler):
    def run_affine(self, img_fixed, img_moving, options):
        # Implementation of affine registration using Greedy
        pass

    def run_deformable(self, img_fixed, img_moving, options):
        # Implementation of deformable registration using Greedy
        pass

    def run_reslice_segmentation(self, img_to_reslice, img_reference, options):
        # Implementation of segmentation reslicing using Greedy
        pass

    def run_reslice_mesh(self, mesh_to_reslice, img_reference, options):
        # Implementation of mesh reslicing using Greedy
        pass


    def run_registration_and_reslice(self, img_fixed, img_moving, img_to_reslice, mesh_to_reslice, options) -> dict:
        return super().run_registration_and_reslice(img_fixed, img_moving, img_to_reslice, mesh_to_reslice, options)


    def get_device_type(self) -> str:
        return "cpu" 