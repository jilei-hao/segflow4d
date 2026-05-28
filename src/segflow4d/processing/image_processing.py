import numpy as np
import SimpleITK as sitk
from segflow4d.utility.image_helper.image_helper_factory import create_image_helper
from segflow4d.common.types.interpolation_type import InterpolationType
from segflow4d.common.types.tp_image_group import TPImageGroup
from segflow4d.common.types.image_wrapper import ImageWrapper

def create_reference_mask(seg_ref_image: ImageWrapper, scale_factor: float, dilation_radius: int) -> ImageWrapper:
    """
    Create a reference mask by resampling the segmentation reference image.

    Args:
        seg_ref_image (sitk.Image): The segmentation reference image.
        scale_factor (float): The scale factor for resampling (0.5 = half resolution, 2.0 = double resolution).
        dilation_radius (int): The radius for binary dilation.

    Returns:
        sitk.Image: The resampled reference mask.
    """

    image_helper = create_image_helper()

    binary_mask = image_helper.binary_threshold(seg_ref_image, lo=1, hi=255)
    dilated_mask = image_helper.binary_dilate(binary_mask, radius=dilation_radius)
    rs_mask = image_helper.resample(dilated_mask, scale_factor=scale_factor, interpolation=InterpolationType.NEAREST)

    return rs_mask

def create_tp_images(image4d: ImageWrapper, target_timepoints: list[int], scale_factor: float) -> dict[int, TPImageGroup]:
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
        tp_image_lowres = image_helper.resample(ImageWrapper(tp_image), scale_factor=scale_factor, interpolation=InterpolationType.LINEAR)
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


def compute_union_bbox(
    masks: list[ImageWrapper],
    padding_voxels: int,
    min_size_voxels: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Compute the union bounding box of all non-zero voxels across a list of masks,
    padded by ``padding_voxels`` on each side and clamped to image extents.

    All masks must share the same grid (size); spacing/direction/origin are not
    checked because the result is expressed in voxel index space.

    Args:
        masks: List of label masks. Any non-zero voxel is treated as foreground.
        padding_voxels: Number of voxels to pad on each side of the union bbox.
        min_size_voxels: If > 0, each output dimension is grown symmetrically
            (clamped to the image extent) so it reaches at least this many
            voxels. If the image itself is smaller than ``min_size_voxels`` in
            some dimension, the result is the full extent for that dimension.

    Returns:
        Tuple ``(start, size)`` in ITK index space (x, y, z order for 3-D inputs).

    Raises:
        ValueError: If ``masks`` is empty, every mask is empty, masks have
            mismatched grids, ``padding_voxels`` is negative, or
            ``min_size_voxels`` is negative.
    """
    if padding_voxels < 0:
        raise ValueError(f"padding_voxels must be >= 0, got {padding_voxels}")
    if min_size_voxels < 0:
        raise ValueError(f"min_size_voxels must be >= 0, got {min_size_voxels}")
    if not masks:
        raise ValueError("masks list is empty")

    reference_size: tuple | None = None
    for i, m in enumerate(masks):
        if m is None or m.get_data() is None:
            raise ValueError(f"masks[{i}] has no image data")
        size = m.get_data().GetSize()
        if reference_size is None:
            reference_size = size
        elif size != reference_size:
            raise ValueError(
                f"masks[{i}] size {size} does not match masks[0] size {reference_size}"
            )

    assert reference_size is not None
    dim = len(reference_size)
    union_lo: list[int] | None = None
    union_hi: list[int] | None = None  # exclusive

    stats = sitk.LabelShapeStatisticsImageFilter()

    for m in masks:
        data = sitk.Cast(m.get_data(), sitk.sitkUInt32)
        stats.Execute(data)
        for label in stats.GetLabels():
            if label == 0:
                continue
            bbox = stats.GetBoundingBox(label)  # (x, y, z, sx, sy, sz)
            lo = list(bbox[:dim])
            sz = list(bbox[dim:2 * dim])
            hi = [lo[k] + sz[k] for k in range(dim)]
            if union_lo is None:
                union_lo, union_hi = lo, hi
            else:
                union_lo = [min(union_lo[k], lo[k]) for k in range(dim)]
                union_hi = [max(union_hi[k], hi[k]) for k in range(dim)]

    if union_lo is None or union_hi is None:
        raise ValueError("every mask in the list is empty (no foreground voxels)")

    start = [max(0, union_lo[k] - padding_voxels) for k in range(dim)]
    end = [min(reference_size[k], union_hi[k] + padding_voxels) for k in range(dim)]

    # Symmetrically grow each dim to reach min_size_voxels, staying inside the
    # image. If symmetric expansion would push past one side, the leftover is
    # added to the other side. If the image itself is shorter than min_size in
    # some dim, that dim ends up at the full image extent.
    if min_size_voxels > 0:
        for k in range(dim):
            cur = end[k] - start[k]
            if cur >= min_size_voxels:
                continue
            need = min_size_voxels - cur
            left = need // 2
            right = need - left
            new_start = start[k] - left
            new_end = end[k] + right
            if new_start < 0:
                new_end += -new_start
                new_start = 0
            if new_end > reference_size[k]:
                new_start -= (new_end - reference_size[k])
                new_end = reference_size[k]
            start[k] = max(0, new_start)
            end[k] = min(reference_size[k], new_end)

    size = [end[k] - start[k] for k in range(dim)]
    return start, size


def crop_to_bbox(image: ImageWrapper, start: list[int], size: list[int]) -> ImageWrapper:
    """
    Extract an index-space region from ``image``.

    Spacing and direction are preserved; the origin shifts so the sub-volume
    retains its physical-space location. No resampling occurs.

    Args:
        image: Input image.
        start: Lower-corner index, length matches image dimension.
        size: Region size in voxels.

    Returns:
        Cropped image.
    """
    data = image.get_data()
    if data is None:
        raise ValueError("image has no data")
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetSize(list(size))
    roi.SetIndex(list(start))
    return ImageWrapper(roi.Execute(data))


def uncrop_to_reference(
    cropped: ImageWrapper,
    reference: ImageWrapper,
    start: list[int],
    fill_value: float = 0,
) -> ImageWrapper:
    """
    Paste a cropped sub-volume back into the reference's full frame.

    The output has the reference's size/spacing/origin/direction and the
    cropped pixel type. Voxels outside the pasted region are set to
    ``fill_value`` (default 0).

    Args:
        cropped: Cropped image (typically produced by ``crop_to_bbox``).
        reference: Image whose grid the output should match.
        start: Lower-corner index where ``cropped`` is pasted into the
            reference frame.
        fill_value: Value for voxels outside the pasted region.

    Returns:
        Image with the reference's grid and the cropped content pasted at
        ``start``.
    """
    cropped_data = cropped.get_data()
    reference_data = reference.get_data()
    if cropped_data is None:
        raise ValueError("cropped image has no data")
    if reference_data is None:
        raise ValueError("reference image has no data")

    ref_size = list(reference_data.GetSize())
    pixel_id = cropped_data.GetPixelID()

    if fill_value == 0:
        dest = sitk.Image(ref_size, pixel_id)
    else:
        # sitk.Image has no scalar-fill constructor — route through numpy.
        zyx_shape = list(reversed(ref_size))
        sample = sitk.GetArrayFromImage(sitk.Image([1] * len(ref_size), pixel_id))
        dest = sitk.GetImageFromArray(np.full(zyx_shape, fill_value, dtype=sample.dtype))

    dest.SetSpacing(reference_data.GetSpacing())
    dest.SetOrigin(reference_data.GetOrigin())
    dest.SetDirection(reference_data.GetDirection())

    paste = sitk.PasteImageFilter()
    paste.SetSourceIndex([0] * len(cropped_data.GetSize()))
    paste.SetSourceSize(list(cropped_data.GetSize()))
    paste.SetDestinationIndex(list(start))
    return ImageWrapper(paste.Execute(dest, cropped_data))