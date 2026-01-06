from registration.abstract_registration_handler import AbstractRegistrationHandler

class FireantsRegistrationHandler(AbstractRegistrationHandler):

    def run_affine(self, img_fixed, img_moving, options):
        # Implementation of affine registration using Fireants
        pass


    def run_deformable(self, img_fixed, img_moving, options):
        # Implementation of deformable registration using Fireants
        pass


    def run_reslice_segmentation(self, img_to_reslice, img_reference, options):
        # Implementation of segmentation reslicing using Fireants
        pass


    def run_reslice_mesh(self, mesh_to_reslice, img_reference, options):
        # Implementation of mesh reslicing using Fireants
        pass


    def run_registration_and_reslice(self, img_fixed, img_moving, img_to_reslice, mesh_to_reslice, options):
        return super().run_registration_and_reslice(img_fixed, img_moving, img_to_reslice, mesh_to_reslice, options)


    def get_device_type(self) -> str:
        return "cuda"  # or "GPU" depending on Fireants capabilities