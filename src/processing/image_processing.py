import SimpleITK as sitk
from utility.image_helper.image_helper_factory import create_image_helper
from common.types.interpolation_type import InterpolationType

def create_reference_mask(seg_ref_image: sitk.Image, resample_factor: float, dilation_radius: int) -> sitk.Image:
    """
    Create a reference mask by resampling the segmentation reference image.

    Args:
        seg_ref_image (sitk.Image): The segmentation reference image.
        resample_factor (float): The factor by which to resample the image.
        dilation_radius (int): The radius for binary dilation.

    Returns:
        sitk.Image: The resampled reference mask.
    """

    image_helper = create_image_helper()

    rs_image = image_helper.resample(seg_ref_image, resample_factor=resample_factor, interpolation=InterpolationType.NEAREST)
    binary_mask = image_helper.binary_threshold(rs_image, lo=1, hi=255)
    dilated_mask = image_helper.binary_dilate(binary_mask, radius=dilation_radius)

    return dilated_mask