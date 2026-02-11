from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from common.types.propagation_options import PropagationOptions
from common.types.tp_data import TPData
from registration.registration_handler.fireants.fireants_registration_options import FireantsRegistrationOptions
from registration import AbstractRegistrationHandler
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
from registration.registration_handler.fireants.gpu_mesh_warper import warp_mesh_vertices

logger = logging.getLogger(__name__)

class FireantsRegistrationHandler(AbstractRegistrationHandler):
    def __init__(self):
        super().__init__()
        logger.info("Initialized FireantsRegistrationHandler")

    def _cleanup_gpu(self):
        """Force GPU memory cleanup for current device only"""
        device_id = torch.cuda.current_device()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_id)  # Sync only current device
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"GPU cleanup warning on cuda:{device_id}: {e}")
        gc.collect()

    def run_affine(self, img_fixed, img_moving, options):
        pass

    def run_deformable(self, img_fixed, img_moving, options):
        pass

    def run_reslice_segmentation(self, img_to_reslice, img_reference, options):
        pass

    def run_reslice_mesh(self, mesh_to_reslice, img_reference, options):
        pass

    def _get_warp_image_from_tensor(self, warp_field: torch.Tensor, reference_image: ImageWrapper) -> ImageWrapper:
        reference_image_data = reference_image.get_data()
        if reference_image_data is None:
            raise ValueError("Reference image data is None when creating warp image")
        
        # Extract numpy array and move to CPU
        warp_field_np = warp_field.detach().cpu().numpy()  # Shape: [N, H, W, D, dims] or [N, H, W, dims]

        # Remove batch dimension if present
        if warp_field_np.ndim == 5:  # 3D: [1, D, H, W, 3]
            warp_field_np = warp_field_np[0]  # [D, H, W, 3]
        elif warp_field_np.ndim == 4:  # 2D: [1, H, W, 2]
            warp_field_np = warp_field_np[0]  # [H, W, 2]

        # Reorder from [D, H, W, 3] to [3, D, H, W] for SimpleITK multi-component format
        warp_field_np = np.moveaxis(warp_field_np, -1, 0)

        # Create SimpleITK image as multi-component (vector image)
        warp_image = sitk.GetImageFromArray(warp_field_np, isVector=True)

        # Copy metadata from fixed image
        warp_image.SetSpacing(reference_image_data.GetSpacing())
        warp_image.SetOrigin(reference_image_data.GetOrigin())
        warp_image.SetDirection(reference_image_data.GetDirection())

        logger.debug(f"Warp image created with shape: {warp_image.GetSize()}, components: {warp_image.GetNumberOfComponentsPerPixel()}")

        return ImageWrapper(warp_image)


    def run_registration_and_reslice(self, img_fixed, img_moving, img_to_reslice, mesh_to_reslice, options: PropagationOptions, mask_fixed=None, mask_moving=None) -> TPData:
        """
        Perform affine + deformable registration and reslice images/segmentations.
        
        Note: options may come as either PropagationOptions or dict due to multiprocessing serialization.
        """
        
        # Get the current device that was already set by caller
        device_id = torch.cuda.current_device()
        device_str = f"cuda:{device_id}"
        logger.info(f"Running registration on device: {device_str}")
        
        # Ensure all operations happen on this device
        torch.cuda.set_device(device_id)

        # Handle both PropagationOptions and dict (from multiprocessing serialization)
        if isinstance(options, dict):
            # options itself is a dict due to multiprocessing
            backend_options_dict = options.get('registration_backend_options', {})
            if isinstance(backend_options_dict, dict):
                backend_options = FireantsRegistrationOptions(**backend_options_dict)
            else:
                backend_options = backend_options_dict
        else:
            # options is a PropagationOptions object
            backend_options = options.registration_backend_options
            if isinstance(backend_options, dict):
                backend_options = FireantsRegistrationOptions(**backend_options)
        
        if not isinstance(backend_options, FireantsRegistrationOptions):
            raise ValueError("Expected FireantsRegistrationOptions or dict for FireantsRegistrationHandler")
        
        # Extract options
        scales = backend_options.scales
        affine_iterations = backend_options.affine_iterations
        deformable_iterations = backend_options.deformable_iterations
        affine_lr = backend_options.affine_lr
        deformable_lr = backend_options.deformable_lr
        loss_type = backend_options.loss_type
        cc_kernel_size = backend_options.cc_kernel_size
        deformation_type = backend_options.deformation_type
        smooth_grad_sigma = backend_options.smooth_grad_sigma
        smooth_warp_sigma = backend_options.smooth_warp_sigma
        
        try:
            # Get ITK images directly
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
            
            # Extract mask ITK images if provided (needed for both stages)
            itk_mask_fixed = mask_fixed.get_data() if mask_fixed is not None else None
            if mask_moving is None and mask_fixed is not None:
                mask_moving = mask_fixed
            itk_mask_moving = mask_moving.get_data() if mask_moving is not None else None
            
            # ==================================================================
            # Stage 1: Affine registration
            # ==================================================================
            logger.info("Starting affine registration...")
            start_affine = time()

            # Create images on specific device
            with torch.cuda.device(device_id):
                fa_image_fixed = Image(itk_fixed, device=device_str)
                fa_image_moving = Image(itk_moving, device=device_str)

                if itk_mask_fixed is not None:
                    fa_mask_fixed = Image(itk_mask_fixed, is_segmentation=True, device=device_str)
                    fa_image_fixed.array = fa_image_fixed.array * fa_mask_fixed.array
                    logger.debug("Applied fixed mask to fixed image")

                if itk_mask_moving is not None:
                    fa_mask_moving = Image(itk_mask_moving, is_segmentation=True, device=device_str)
                    fa_image_moving.array = fa_image_moving.array * fa_mask_moving.array
                    logger.debug("Applied moving mask to moving image")
                
                batch_fixed = BatchedImages(fa_image_fixed)
                batch_moving = BatchedImages(fa_image_moving)
                
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
            
            # Clean up affine stage objects
            logger.debug("Deleting affine stage objects...")
            del affine_reg
            
            # Minimal cleanup - don't sync across devices
            gc.collect()
            torch.cuda.empty_cache()

            # ==================================================================
            # Stage 2: Deformable registration with affine initialization
            # ==================================================================
            logger.info("Starting deformable registration...")
            start_deformable = time()
            resliced_seg_mesh = None
            
            with torch.cuda.device(device_id):
                # Recreate batch images for deformable stage to ensure device consistency
                fa_image_fixed_def = Image(itk_fixed, device=device_str)
                fa_image_moving_def = Image(itk_moving, device=device_str)
                
                # Re-apply masks if needed
                if itk_mask_fixed is not None:
                    fa_mask_fixed_def = Image(itk_mask_fixed, is_segmentation=True, device=device_str)
                    fa_image_fixed_def.array = fa_image_fixed_def.array * fa_mask_fixed_def.array
                if itk_mask_moving is not None:
                    fa_mask_moving_def = Image(itk_mask_moving, is_segmentation=True, device=device_str)
                    fa_image_moving_def.array = fa_image_moving_def.array * fa_mask_moving_def.array
                
                batch_fixed_def = BatchedImages(fa_image_fixed_def)
                batch_moving_def = BatchedImages(fa_image_moving_def)

                fa_image_to_reslice = Image(itk_to_reslice, is_segmentation=False, device=device_str)
                batch_to_reslice = BatchedImages([fa_image_to_reslice])
                batch_to_reslice.interpolate_mode = 'nearest'
                
                affine_matrix_gpu = affine_matrix.to(device_str)
                
                deformable_reg = GreedyRegistration(
                    scales=scales,
                    iterations=deformable_iterations,
                    fixed_images=batch_fixed_def,
                    moving_images=batch_moving_def,
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
                
                # Reslice target image
                logger.info("Reslicing target image...")
                moved_resliced = deformable_reg.evaluate(batch_fixed_def, batch_to_reslice)
                logger.debug(f"Interpolator type: {batch_to_reslice.get_interpolator_type()}")
                
                # Extract data immediately and copy to CPU
                resliced_tensor = moved_resliced[0].detach().cpu().numpy().copy()

                # Get Warp Image
                mesh_warp_field = deformable_reg.get_inverse_warped_coordinates(batch_fixed_def, batch_moving_def, None)
                mesh_warp_image = self._get_warp_image_from_tensor(mesh_warp_field, img_fixed)

                # Reslice mesh
                if mesh_to_reslice is not None:
                    logger.info("Reslicing target mesh...")
                    mesh_vertices = mesh_to_reslice.get_vertices()  # numpy array (N, 3)
                    mesh_vertices_tensor = torch.from_numpy(mesh_vertices).to(device_str, dtype=torch.float32)
                    warped_vertices = warp_mesh_vertices(
                        mesh_vertices_tensor,
                        mesh_warp_field,
                        img_fixed,
                        img_moving
                    )
                    warped_vertices_np = warped_vertices.cpu().detach().numpy()
                    resliced_seg_mesh = mesh_to_reslice.update_vertices(warped_vertices_np)
            
            # Clean up deformable stage
            logger.debug("Deleting deformable stage objects...")
            del deformable_reg
            del batch_to_reslice
            del batch_fixed_def
            del batch_moving_def
            del fa_image_to_reslice
            del moved_resliced
            del affine_matrix_gpu
            # Also delete affine stage objects if still around
            del batch_fixed
            del batch_moving
            
            gc.collect()
            torch.cuda.empty_cache()
            
            # Convert resliced image back to ITK format
            logger.debug("Full shape of resliced tensor: " + str(resliced_tensor.shape))
            
            # resliced_tensor shape is (1, D, H, W) for non-segmentation mode
            # Squeeze the channel dimension and convert to integer labels
            if resliced_tensor.ndim == 4:
                resliced_labels = resliced_tensor[0]  # Remove channel dim: (D, H, W)
            else:
                resliced_labels = resliced_tensor
            
            # Round to nearest integer and cast to appropriate type
            # Nearest neighbor interpolation should preserve values, but floating point
            # precision may introduce small errors
            resliced_labels = np.round(resliced_labels).astype(np.uint8)
            
            logger.debug(f"Resliced labels shape: {resliced_labels.shape}, "
                        f"unique values: {np.unique(resliced_labels)}")
            
            resliced_itk = sitk.GetImageFromArray(resliced_labels)
            resliced_itk.SetSpacing(reslice_meta['spacing'])
            resliced_itk.SetOrigin(reslice_meta['origin'])
            resliced_itk.SetDirection(reslice_meta['direction'])
            
            # Mesh reslicing to be implemented
            resliced_meshes = dict[str, MeshWrapper]()
            if mesh_to_reslice is not None:
                logger.warning("Mesh reslicing not yet implemented")

            # if resliced_seg_mesh is None:
            #     raise RuntimeError("Resliced segmentation mesh is None after reslicing")
            
            logger.info("Registration and reslicing completed successfully")
            
            return TPData(
                affine_matrix=affine_matrix.numpy().copy(),
                resliced_image=ImageWrapper(resliced_itk),
                resliced_segmentation_mesh=resliced_seg_mesh,
                resliced_meshes=resliced_meshes,
                warp_image=mesh_warp_image
            )
        
        except Exception as e:
            logger.error(f"Registration failed on cuda:{device_id}: {e}", exc_info=True)
            self._cleanup_gpu()
            raise

    def get_device_type(self) -> str:
        return "cuda"
