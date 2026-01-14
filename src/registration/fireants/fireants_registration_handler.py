from common.types.image_wrapper import ImageWrapper
from registration.abstract_registration_handler import AbstractRegistrationHandler
import logging
import torch
from time import time
from fireants.io import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)

# Import threading for thread-safe operations
import threading

# Global lock for thread-safe FireANTs Image creation
_fireants_creation_lock = threading.Lock()

class FireantsRegistrationHandler(AbstractRegistrationHandler):
    def __init__(self):
        super().__init__()
        logger.info("Initialized FireantsRegistrationHandler")

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
        """
        
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
        smooth_grad_sigma = options.get('smooth_grad_sigma', 3.0)
        smooth_warp_sigma = options.get('smooth_warp_sigma', 1.5)
        
        # Initialize variables for cleanup in finally block
        fa_image_fixed = None
        fa_image_moving = None
        fa_image_to_reslice = None
        affine_reg = None
        deformable_reg = None
        batch_fixed = None
        batch_moving = None
        batch_to_reslice = None
        moved_affine = None
        moved_resliced = None
        
        try:
            # Force initial cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Thread-safe image data conversion to avoid lazy tensor issues
            logger.info("Converting images to FireANTs format with thread safety...")
            
            # Get image data and ensure it's materialized (not lazy)
            # Create deep copies to avoid shared state between threads
            import numpy as np
            fixed_data = img_fixed.get_data()
            moving_data = img_moving.get_data()
            reslice_data = img_to_reslice.get_data()
            
            # Ensure data is contiguous and create copies to avoid shared memory
            if hasattr(fixed_data, 'copy'):
                fixed_data = fixed_data.copy()
            if hasattr(moving_data, 'copy'):
                moving_data = moving_data.copy()
            if hasattr(reslice_data, 'copy'):
                reslice_data = reslice_data.copy()
            
            # Ensure any PyTorch tensors are materialized (not lazy)
            if hasattr(torch, 'is_tensor'):
                if torch.is_tensor(fixed_data) and hasattr(fixed_data, '_base'):
                    fixed_data = fixed_data.detach().cpu().numpy().copy()
                if torch.is_tensor(moving_data) and hasattr(moving_data, '_base'):
                    moving_data = moving_data.detach().cpu().numpy().copy()
                if torch.is_tensor(reslice_data) and hasattr(reslice_data, '_base'):
                    reslice_data = reslice_data.detach().cpu().numpy().copy()
            
            # Convert to FireANTs format with thread synchronization


            
            with _fireants_creation_lock:
                fa_image_fixed = Image(fixed_data)
                # Small delay to prevent rapid concurrent access
                import time as time_module
                time_module.sleep(0.01)
                
            with _fireants_creation_lock:
                fa_image_moving = Image(moving_data)
                time_module.sleep(0.01)
                
            with _fireants_creation_lock:
                fa_image_to_reslice = Image(reslice_data, is_segmentation=True)

            # Prepare loss kwargs
            loss_kwargs = {}
            if loss_type == 'cc':
                loss_kwargs['cc_kernel_size'] = cc_kernel_size
            
            # ==================================================================
            # Stage 1: Affine registration
            # ==================================================================
            logger.info("Starting affine registration...")
            start_affine = time()
            
            batch_fixed = BatchedImages([fa_image_fixed])
            batch_moving = BatchedImages([fa_image_moving])
            
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
            
            # Get affine matrix - keep on GPU for deformable registration
            affine_matrix = affine_reg.get_affine_matrix().detach().clone()
            
            # Aggressive cleanup after affine stage
            del moved_affine
            moved_affine = None
            del affine_reg
            affine_reg = None
            del batch_fixed
            del batch_moving
            batch_fixed = None
            batch_moving = None
            
            # Force garbage collection and GPU cleanup
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # ==================================================================
            # Stage 2: Deformable registration with affine initialization
            # ==================================================================
            logger.info("Starting deformable registration...")
            start_deformable = time()
            
            # Create new BatchedImages wrappers
            batch_fixed = BatchedImages([fa_image_fixed])
            batch_moving = BatchedImages([fa_image_moving])
            batch_to_reslice = BatchedImages([fa_image_to_reslice])
            
            deformable_reg = GreedyRegistration(
                scales=scales,
                iterations=deformable_iterations,
                fixed_images=batch_fixed,
                moving_images=batch_moving,
                deformation_type=deformation_type,
                smooth_grad_sigma=smooth_grad_sigma,
                smooth_warp_sigma=smooth_warp_sigma,
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
            
            # Extract data immediately and copy to CPU
            resliced_tensor = moved_resliced[0].detach().cpu().numpy().copy()
            
            # Convert resliced image back to original format
            # Handle segmentation (threshold) vs continuous image
            if fa_image_to_reslice.itk_image is not None:
                itk_original = fa_image_to_reslice.itk_image
                
                if resliced_tensor.shape[0] == 1:
                    # Binary segmentation - threshold at 0.5
                    resliced_labels = (resliced_tensor[0] > 0.5).astype('uint8')
                else:
                    # Multi-class - take argmax
                    resliced_labels = np.argmax(resliced_tensor, axis=0).astype('uint8')
                
                resliced_itk = sitk.GetImageFromArray(resliced_labels)
                resliced_itk.SetSpacing(itk_original.GetSpacing())
                resliced_itk.SetOrigin(itk_original.GetOrigin())
                resliced_itk.SetDirection(itk_original.GetDirection())
            else:
                # Continuous image - use as is
                resliced_itk = sitk.GetImageFromArray(resliced_tensor[0] if resliced_tensor.shape[0] == 1 else resliced_tensor)
            
            # Mesh reslicing to be implemented
            resliced_mesh = None
            if mesh_to_reslice is not None:
                logger.warning("Mesh reslicing not yet implemented")
            
            logger.info("Registration and reslicing completed successfully")
            
            return {
                'affine_matrix': affine_matrix,
                'resliced_image': ImageWrapper(resliced_itk),
                'resliced_mesh': resliced_mesh
            }
        
        finally:
            # Comprehensive cleanup - delete everything explicitly
            if moved_resliced is not None:
                del moved_resliced
            if deformable_reg is not None:
                del deformable_reg
            if moved_affine is not None:
                del moved_affine
            if affine_reg is not None:
                del affine_reg
            if batch_to_reslice is not None:
                del batch_to_reslice
            if batch_fixed is not None:
                del batch_fixed
            if batch_moving is not None:
                del batch_moving
            if fa_image_to_reslice is not None:
                del fa_image_to_reslice
            if fa_image_moving is not None:
                del fa_image_moving
            if fa_image_fixed is not None:
                del fa_image_fixed
                
            # Force cleanup
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            logger.debug("FireANTs registration cleanup completed")

    def get_device_type(self) -> str:
        return "cuda"  # or "GPU" depending on Fireants capabilities