from processing.image_processing import create_reference_mask

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create reference mask from image")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--resample-rate", type=float, required=True, help="Resample rate")
    parser.add_argument("--dilation-radius", type=int, required=True, help="Dilation radius")
    args = parser.parse_args()

    import SimpleITK as sitk
    input_image = sitk.ReadImage(args.input)
    reference_mask = create_reference_mask(
        seg_ref_image=input_image,
        resample_factor=args.resample_rate,
        dilation_radius=args.dilation_radius
    )
    sitk.WriteImage(reference_mask, args.output)


if __name__ == "__main__":
    main()