from ..image_processing import create_tp_images

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create TP images from input image")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output images")
    parser.add_argument("--resample-factor", type=float, required=True, help="Resample factor")
    parser.add_argument("--timepoints", type=int, nargs='+', required=True, help="List of timepoints to extract")
    args = parser.parse_args()

    import SimpleITK as sitk
    input_image = sitk.ReadImage(args.input)
    tp_images = create_tp_images(
        image4d=input_image,
        target_timepoints=args.timepoints,
        resample_factor=args.resample_factor
    )
    for t, tp_image_group in tp_images.items():
        sitk.WriteImage(tp_image_group.image_fullres.get_data(), f"{args.output_dir}/i3-fullres_tp-{t:03d}.nii.gz")
        sitk.WriteImage(tp_image_group.image_lowres.get_data(), f"{args.output_dir}/i3-lowres_tp-{t:03d}.nii.gz")