from utility.image_helper.image_helper_factory import create_image_helper
from common.types.interpolation_type import InterpolationType

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Resample image to reference image")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--output", required=True, help="Path to output image")
    args = parser.parse_args()

    import SimpleITK as sitk
    input_image = sitk.ReadImage(args.input)
    reference_image = sitk.ReadImage(args.reference)

    image_helper = create_image_helper()
    resampled_image = image_helper.resample_to_reference(
        image=input_image,
        reference_image=reference_image,
        interpolation=InterpolationType.NEAREST
    )
    
    sitk.WriteImage(resampled_image, args.output)


if __name__ == "__main__":
    main()