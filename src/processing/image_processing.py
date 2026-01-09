import SimpleITK as sitk
from utility.image_helper.image_helper_factory import create_image_helper
from common.types.interpolation_type import InterpolationType
from common.types.tp_image_group import TPImageGroup
from common.types.image_wrapper import ImageWrapper

def create_reference_mask(seg_ref_image: ImageWrapper, resample_factor: float, dilation_radius: int) -> ImageWrapper:
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

def create_tp_images(image4d: ImageWrapper, target_timepoints: list[int], resample_factor: float) -> dict[int, TPImageGroup]:
    """
    Generate 3D images for specified timepoints from a 4D image.

    Args:
        image4d (ImageWrapper): The input 4D image.
        target_timepoints (list[int]): List of timepoints to extract.

    Returns:
        dict[int, TPImageGroup]: A dictionary mapping timepoints to their corresponding 3D image groups.
    """

    tp_images = {}  
    image_helper = create_image_helper()

    for t in target_timepoints:
        print(f"Extracting timepoint {t}...")
        extractor = sitk.ExtractImageFilter()
        size = list(image4d.get_data().GetSize())
        size[3] = 0  # Extract along the time dimension
        index = [0, 0, 0, t - 1]  # Timepoint index (0-based)
        extractor.SetSize(size)
        extractor.SetIndex(index)
        tp_image = extractor.Execute(image4d.get_data())
        tp_image_lowres = image_helper.resample(ImageWrapper(tp_image), resample_factor=resample_factor, interpolation=InterpolationType.LINEAR)
        tp_images[t] = TPImageGroup(image_fullres=ImageWrapper(tp_image), image_lowres=tp_image_lowres)

    return tp_images


def create_high_res_mask(ref_seg_image: ImageWrapper, low_res_mask: ImageWrapper) -> ImageWrapper:
    """
    Create a high-resolution mask by resampling the low-resolution mask to the reference segmentation image.

    Args:
        ref_seg_image (ImageWrapper): The reference segmentation image.
        low_res_mask (ImageWrapper): The low-resolution mask image.

    Returns:
        ImageWrapper: The high-resolution mask image.
    """

    image_helper = create_image_helper()
    high_res_mask = image_helper.resample_to_reference(low_res_mask, ref_seg_image, interpolation=InterpolationType.NEAREST)
    return high_res_mask