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
    parser.add_argument("--seg-to-warp", type=str, help="/path/to/image/to/warp")
    parser.add_argument("--output-warped", type=str, default="warped_image.nii.gz", help="Output path for warped image")
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
    seg_to_warp = None
    if args.seg_to_warp is not None:
        seg_to_warp = Image.load_file(args.seg_to_warp, is_segmentation=False) # turning off segmentation for large multi-label images
    
    # Batchify them (we only have a single image per batch, but we can pass multiple images)
    batch1 = BatchedImages([image1])
    batch2 = BatchedImages([image2])
    
    seg_batch_to_warp = batch2
    if seg_to_warp is not None:
        seg_batch_to_warp = BatchedImages([seg_to_warp])
        seg_batch_to_warp.interpolate_mode = 'nearest'  # for multi-label segmentation

    
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
    seg_affine_warped = affine.evaluate(batch1, seg_batch_to_warp)
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
    moved = reg.evaluate(batch1, seg_batch_to_warp)

    # Save moved image
    print(f"\nSaving moved image to {args.output_warped} ...")
    
    if seg_to_warp is not None:
        # For segmentation, the interpolator type is set to 'nearest'
        # This is done automatically by is_segmentation=True
        seg_resliced = reg.evaluate(batch1, seg_batch_to_warp)
        
        # Convert tensor back to SimpleITK image manually
        import numpy as np
        
        resliced_tensor = seg_resliced[0].detach().cpu().numpy()
        
        if resliced_tensor.ndim == 4:
            moved_labels = resliced_tensor[0]
        else:
            moved_labels = resliced_tensor

        resliced_labels = np.round(moved_labels).astype(np.uint8)
        
        itk_moved = sitk.GetImageFromArray(resliced_labels)
        itk_moved.SetSpacing(seg_to_warp.itk_image.GetSpacing())
        itk_moved.SetOrigin(seg_to_warp.itk_image.GetOrigin())
        itk_moved.SetDirection(seg_to_warp.itk_image.GetDirection())
        
        sitk.WriteImage(itk_moved, args.output_warped)
    else:
        reg.save_moved_images(moved, args.output_warped)
        

if __name__ == "__main__":
    main() 
