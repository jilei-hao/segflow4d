#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
from time import time
from fireants.io import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.interpolator import fireants_interpolator
from fireants.utils.imageutils import jacobian

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run FireANTs registration")
    parser.add_argument("--input-fix", type=str, required=True, help="/path/to/fix/image")
    parser.add_argument("--input-mov", type=str, required=True, help="/path/to/mov/image")
    parser.add_argument("--fix-mask", type=str, help="/path/to/fix/mask")
    parser.add_argument("--mov-mask", type=str, help="/path/to/mov/mask")
    parser.add_argument("--input-to-warp", type=str, help="/path/to/image/to/warp")
    parser.add_argument("--output-warp", type=str, default="warped_image.nii.gz", help="Output path for warped image")
    args = parser.parse_args()


    # Load the images
    print("Loading images...")
    image1 = Image.load_file(args.input_fix)
    image2 = Image.load_file(args.input_mov)


    # Load masks if provided
    if args.fix_mask is not None:
        fix_mask = Image.load_file(args.fix_mask, is_segmentation=True)
        if args.mov_mask is None:
            mov_mask = fix_mask
        else:
            mov_mask = Image.load_file(args.mov_mask, is_segmentation=True)

        image1.array = image1.array * fix_mask.array
        image2.array = image2.array * mov_mask.array
    else:
        fix_mask = None
        

    # load image to warp if provided
    image_to_warp = None
    if args.input_to_warp is not None:
        image_to_warp = Image.load_file(args.input_to_warp, is_segmentation=True)
    
    # Batchify them (we only have a single image per batch, but we can pass multiple images)
    batch1 = BatchedImages([image1])
    batch2 = BatchedImages([image2])
    
    batch_to_warp = batch2
    if image_to_warp is not None:
        batch_to_warp = BatchedImages([image_to_warp])

    
    # Check device name
    print(f"Using device: {batch1().device}")
    
    # Print coordinate transformation matrices
    # print("\nCoordinate transformation matrices:")
    # print("phy2torch:")
    # print(image1.phy2torch)
    # print("\ntorch2phy:")
    # print(image1.torch2phy)
    # print("\nphy2torch @ torch2phy:")
    # print(image1.phy2torch @ image1.torch2phy)
    
    # Perform affine registration
    print("\nPerforming affine registration...")
    scales = [4.0, 2.0, 1.0]  # scales at which to perform registration
    iterations = [200, 100, 50]
    optim = 'Adam'
    lr = 3e-3
    
    # Create affine registration object
    affine = AffineRegistration(scales, iterations, batch1, batch2, 
                                loss_type='mse',
                                # loss_type='cc', cc_kernel_size=3,
                                optimizer=optim, optimizer_lr=lr)
    
    # Run registration
    start = time()
    affine.optimize()
    moved = affine.evaluate(batch1, batch_to_warp)
    end = time()
    print(f"Affine registration runtime: {end - start:.2f} seconds")

        # Get affine matrix BEFORE deleting
    affine_matrix = affine.get_affine_matrix().detach().clone()
    
    # Release affine registration memory
    del affine
    torch.cuda.empty_cache()
    
    # Perform deformable registration
    print("\nPerforming deformable registration...")
    reg = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 100, 25],
                            fixed_images=batch1, moving_images=batch2,
                            deformation_type='compositive',
                            smooth_grad_sigma=1.5,
                            loss_type='mse',
                            # loss_type='cc', cc_kernel_size=3,
                            optimizer='adam', optimizer_lr=0.5,
                            init_affine=affine_matrix)
    
    # Run deformable registration
    start = time()
    reg.optimize()
    end = time()
    print(f"Deformable registration runtime: {end - start:.2f} seconds")
    
    # Get moved image
    moved = reg.evaluate(batch1, batch_to_warp)

    # Save moved image
    print(f"\nSaving moved image to {args.output_warp} ...")
    
    if image_to_warp is not None:
        # For segmentation, the interpolator type is set to 'nearest'
        # This is done automatically by is_segmentation=True
        batch_to_warp = BatchedImages([image_to_warp])
        moved = reg.evaluate(batch1, batch_to_warp)
        
        # Convert tensor back to SimpleITK image manually
        import numpy as np
        
        moved_tensor = moved[0].detach()
        moved_np = moved_tensor.cpu().numpy()
        
        if moved_tensor.shape[0] == 1:
            # Binary segmentation - threshold at 0.5
            moved_labels = (moved_np[0] > 0.5).astype('uint8')
        else:
            # Multi-class - take argmax
            moved_labels = np.argmax(moved_np, axis=0).astype('uint8')
        
        itk_moved = sitk.GetImageFromArray(moved_labels)
        itk_moved.SetSpacing(image_to_warp.itk_image.GetSpacing())
        itk_moved.SetOrigin(image_to_warp.itk_image.GetOrigin())
        itk_moved.SetDirection(image_to_warp.itk_image.GetDirection())
        
        sitk.WriteImage(itk_moved, args.output_warp)
    else:
        reg.save_moved_images(moved, args.output_warp)
        

if __name__ == "__main__":
    main() 
