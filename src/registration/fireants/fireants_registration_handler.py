from common.types.image_wrapper import ImageWrapper
from registration.abstract_registration_handler import AbstractRegistrationHandler
import logging
import torch
import gc
from time import time
from fireants.io import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
import SimpleITK as sitk
import numpy as np
from utility.image_helper.image_helper_factory import create_image_helper

logger = logging.getLogger(__name__)

class FireantsRegistrationHandler(AbstractRegistrationHandler):
    def __init__(self):
        super().__init__()
        logger.info("Initialized FireantsRegistrationHandler")

    def _cleanup_gpu(self):
        """Force GPU memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def run_affine(self, img_fixed, img_moving, options):
        pass

    def run_deformable(self, img_fixed, img_moving, options):
        pass

    def run_reslice_segmentation(self, img_to_reslice, img_reference, options):
        pass

    def run_reslice_mesh(self, mesh_to_reslice, img_reference, options):
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
        
        try:
            # Get ITK images directly - ImageWrapper.deepcopy() already created independent copies
            logger.debug("Converting images to FireANTs format...")
            itk_fixed = img_fixed.get_data()
            itk_moving = img_moving.get_data()
            itk_to_reslice = img_to_reslice.get_data()

            ih = create_image_helper()
            unique_labels = ih.get_unique_labels(img_to_reslice)
            logger.debug(f"Unique labels in image to reslice: {unique_labels}")
            
            # Store metadata for result reconstruction
            reslice_meta = {
                'spacing': itk_to_reslice.GetSpacing(),
                'origin': itk_to_reslice.GetOrigin(),
                'direction': itk_to_reslice.GetDirection()
            }
            
            logger.debug(f"Fixed size: {itk_fixed.GetSize()}, Moving size: {itk_moving.GetSize()}, "
                        f"Reslice size: {itk_to_reslice.GetSize()}")
            
            # Prepare loss kwargs
            loss_kwargs = {}
            if loss_type == 'cc':
                loss_kwargs['cc_kernel_size'] = cc_kernel_size
            
            # ==================================================================
            # Stage 1: Affine registration
            # ==================================================================
            logger.info("Starting affine registration...")
            start_affine = time()
            
            # Create FireANTs images for affine stage (pass ITK images)
            fa_image_fixed = Image(itk_fixed)
            fa_image_moving = Image(itk_moving)
            
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
            
            # Get affine matrix and move to CPU immediately
            affine_matrix = affine_reg.get_affine_matrix().detach().cpu().clone()
            
            end_affine = time()
            logger.info(f"Affine registration completed in {end_affine - start_affine:.2f} seconds")
            
            # CRITICAL: Clean up ALL affine stage objects before deformable
            del affine_reg
            del batch_fixed
            del batch_moving
            del fa_image_fixed
            del fa_image_moving
            
            # Force GPU memory cleanup between stages
            logger.debug("Cleaning GPU memory between affine and deformable stages...")
            self._cleanup_gpu()

            # ==================================================================
            # Stage 2: Deformable registration with affine initialization
            # ==================================================================
            logger.info("Starting deformable registration...")
            start_deformable = time()
            
            # Create NEW FireANTs images for deformable stage (pass ITK images)
            fa_image_fixed = Image(itk_fixed)
            fa_image_moving = Image(itk_moving)
            fa_image_to_reslice = Image(itk_to_reslice, is_segmentation=True, background_seg_label=-1)
            
            batch_fixed = BatchedImages([fa_image_fixed])
            batch_moving = BatchedImages([fa_image_moving])
            batch_to_reslice = BatchedImages([fa_image_to_reslice])
            
            # Move affine matrix back to GPU for initialization
            affine_matrix_gpu = affine_matrix.to(torch.cuda.current_device())
            
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
                init_affine=affine_matrix_gpu,
                **loss_kwargs
            )
            
            deformable_reg.optimize()
            end_deformable = time()
            logger.info(f"Deformable registration completed in {end_deformable - start_deformable:.2f} seconds")
            
            # Stage 3: Reslice target image using deformation
            logger.info("Reslicing target image...")
            moved_resliced = deformable_reg.evaluate(batch_fixed, batch_to_reslice)
            logger.debug(f"Interpolator type: {batch_to_reslice.get_interpolator_type()}")
            
            # Extract data immediately and copy to CPU
            resliced_tensor = moved_resliced[0].detach().cpu().numpy().copy()
            
            # Clean up deformable stage
            del deformable_reg
            del batch_fixed
            del batch_moving
            del batch_to_reslice
            del fa_image_fixed
            del fa_image_moving
            del fa_image_to_reslice
            del moved_resliced
            del affine_matrix_gpu
            
            self._cleanup_gpu()
            
            # Convert resliced image back to ITK format using saved metadata
            logger.debug("Full shape of resliced tensor: " + str(resliced_tensor.shape))
            
            # Create output image filled with background (0)
            output_shape = itk_to_reslice.GetSize()[::-1]  # ITK uses (x, y, z), numpy uses (z, y, x)
            resliced_labels = np.zeros(output_shape, dtype=np.uint8)
            
            # Convert resliced image back to ITK format using saved metadata
            if resliced_tensor.shape[0] == 1:
                # Binary segmentation - threshold at 0.5
                resliced_labels = (resliced_tensor[0] > 0.5).astype('uint8')
            else:
                # Multi-class - take argmax
                resliced_labels = np.argmax(resliced_tensor, axis=0).astype('uint8')
            
            resliced_itk = sitk.GetImageFromArray(resliced_labels)
            resliced_itk.SetSpacing(reslice_meta['spacing'])
            resliced_itk.SetOrigin(reslice_meta['origin'])
            resliced_itk.SetDirection(reslice_meta['direction'])

            
            # Mesh reslicing to be implemented
            resliced_mesh = None
            if mesh_to_reslice is not None:
                logger.warning("Mesh reslicing not yet implemented")
            
            logger.info("Registration and reslicing completed successfully")
            
            return {
                'affine_matrix': affine_matrix.numpy().copy(),
                'resliced_image': ImageWrapper(resliced_itk),
                'resliced_mesh': resliced_mesh
            }
        
        except Exception as e:
            logger.error(f"Registration failed: {e}", exc_info=True)
            self._cleanup_gpu()
            raise

    def get_device_type(self) -> str:
        return "cuda"