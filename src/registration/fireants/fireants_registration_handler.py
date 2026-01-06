from registration.abstract_registration_handler import AbstractRegistrationHandler

import logging
logger = logging.getLogger(__name__)

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


    def run_registration_and_reslice(self, img_fixed, img_moving, img_to_reslice, mesh_to_reslice, options) -> dict:
        """
        Perform affine + deformable registration and reslice images/segmentations.
        
        Args:
            img_fixed: Fixed reference image
            img_moving: Moving image to register
            img_to_reslice: Image/segmentation to warp using the computed deformation
            mesh_to_reslice: Mesh to reslice (to be implemented)
            options: Dictionary containing registration parameters
                - scales: List of scales for multi-scale registration [default: [4.0, 2.0, 1.0]]
                - affine_iterations: Iterations for affine stage [default: [200, 100, 50]]
                - deformable_iterations: Iterations for deformable stage [default: [200, 100, 25]]
                - affine_lr: Learning rate for affine optimizer [default: 3e-3]
                - deformable_lr: Learning rate for deformable optimizer [default: 0.5]
                - loss_type: Loss function type 'mse' or 'cc' [default: 'mse']
                - cc_kernel_size: Kernel size for CC loss [default: 3]
                - deformation_type: 'compositive' or 'additive' [default: 'compositive']
                - smooth_grad_sigma: Smoothing sigma for gradients [default: 1.5]
        
        Returns:
            Dictionary with keys:
                - 'affine_matrix': Computed affine transformation matrix
                - 'resliced_image': Warped img_to_reslice using deformation
                - 'moved_moving': Moving image warped to fixed space
        """
        import torch
        from time import time
        from fireants.io import Image, BatchedImages
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.greedy import GreedyRegistration
        import SimpleITK as sitk
        import numpy as np
        
        logger.info("Starting registration and reslicing with FireANTs")
        
        # Extract options with defaults
        scales = options.get('scales', [4.0, 2.0, 1.0])
        affine_iterations = options.get('affine_iterations', [200, 100, 50])
        deformable_iterations = options.get('deformable_iterations', [200, 100, 25])
        affine_lr = options.get('affine_lr', 3e-3)
        deformable_lr = options.get('deformable_lr', 0.5)
        loss_type = options.get('loss_type', 'mse')
        cc_kernel_size = options.get('cc_kernel_size', 3)
        deformation_type = options.get('deformation_type', 'compositive')
        smooth_grad_sigma = options.get('smooth_grad_sigma', 1.5)
        
        # Convert input images to FireANTs format
        logger.info("Converting images to FireANTs format...")
        batch_fixed = BatchedImages([img_fixed])
        batch_moving = BatchedImages([img_moving])
        batch_to_reslice = BatchedImages([img_to_reslice])
        
        # Prepare loss kwargs
        loss_kwargs = {}
        if loss_type == 'cc':
            loss_kwargs['cc_kernel_size'] = cc_kernel_size
        
        # Stage 1: Affine registration
        logger.info("Starting affine registration...")
        start_affine = time()
        
        affine_reg = AffineRegistration(
            scales=scales,
            iterations=affine_iterations,
            fixed_images=batch_fixed,
            moving_images=batch_moving,
            loss_type=loss_type,
            optimizer='Adam',
            optimizer_lr=affine_lr,
            **loss_kwargs
        )
        
        affine_reg.optimize()
        moved_affine = affine_reg.evaluate(batch_fixed, batch_moving)
        end_affine = time()
        logger.info(f"Affine registration completed in {end_affine - start_affine:.2f} seconds")
        
        # Get affine matrix before cleanup
        affine_matrix = affine_reg.get_affine_matrix().detach().clone()
        del affine_reg
        torch.cuda.empty_cache()
        
        # Stage 2: Deformable registration with affine initialization
        logger.info("Starting deformable registration...")
        start_deformable = time()
        
        deformable_reg = GreedyRegistration(
            scales=scales,
            iterations=deformable_iterations,
            fixed_images=batch_fixed,
            moving_images=batch_moving,
            deformation_type=deformation_type,
            smooth_grad_sigma=smooth_grad_sigma,
            loss_type=loss_type,
            optimizer='adam',
            optimizer_lr=deformable_lr,
            init_affine=affine_matrix,
            **loss_kwargs
        )
        
        deformable_reg.optimize()
        end_deformable = time()
        logger.info(f"Deformable registration completed in {end_deformable - start_deformable:.2f} seconds")
        
        # Stage 3: Reslice target image using deformation
        logger.info("Reslicing target image...")
        moved_resliced = deformable_reg.evaluate(batch_fixed, batch_to_reslice)
        
        # Convert resliced image back to original format
        resliced_tensor = moved_resliced[0].detach()
        resliced_np = resliced_tensor.cpu().numpy()
        
        # Handle segmentation (threshold) vs continuous image
        if img_to_reslice.itk_image is not None:
            itk_original = img_to_reslice.itk_image
            
            if resliced_tensor.shape[0] == 1:
                # Binary segmentation - threshold at 0.5
                resliced_labels = (resliced_np[0] > 0.5).astype('uint8')
            else:
                # Multi-class - take argmax
                resliced_labels = np.argmax(resliced_np, axis=0).astype('uint8')
            
            resliced_itk = sitk.GetImageFromArray(resliced_labels)
            resliced_itk.SetSpacing(itk_original.GetSpacing())
            resliced_itk.SetOrigin(itk_original.GetOrigin())
            resliced_itk.SetDirection(itk_original.GetDirection())
        else:
            # Continuous image - use as is
            resliced_itk = sitk.GetImageFromArray(resliced_np[0] if resliced_np.shape[0] == 1 else resliced_np)
        
        # Mesh reslicing to be implemented
        resliced_mesh = None
        if mesh_to_reslice is not None:
            logger.warning("Mesh reslicing not yet implemented")
        
        logger.info("Registration and reslicing completed successfully")
        
        return {
            'affine_matrix': affine_matrix,
            'resliced_image': resliced_itk,
            'resliced_mesh': resliced_mesh,
            'moved_moving': moved_affine[0].detach().cpu().numpy()
        }


    def get_device_type(self) -> str:
        return "cuda"  # or "GPU" depending on Fireants capabilities