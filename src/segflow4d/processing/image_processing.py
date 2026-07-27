import math
import numpy as np
import SimpleITK as sitk
from segflow4d.utility.image_helper.image_helper_factory import create_image_helper
from segflow4d.common.types.interpolation_type import InterpolationType
from segflow4d.common.types.tp_image_group import TPImageGroup
from segflow4d.common.types.image_wrapper import ImageWrapper

# FireANTs clamps each registration scale level to at least this many voxels per
# spatial dimension (its internal MIN_IMG_SIZE). When an image is smaller it is
# silently upsampled — and the fixed and moving images can be upsampled
# inconsistently, producing a grid/shape mismatch in the loss
# (e.g. "size of tensor a (32) must match the size of tensor b (28)").
# Keep every spatial dim at/above this floor before handing images to FireANTs.
FIREANTS_MIN_IMG_SIZE = 32


def clamp_scale_factor_for_min_size(
    image_size: tuple[int, ...] | list[int],
    scale_factor: float,
    min_size_voxels: int = FIREANTS_MIN_IMG_SIZE,
) -> float:
    """
    Raise ``scale_factor`` just enough that resampling an image of ``image_size``
    never produces a spatial dimension below ``min_size_voxels``.

    Resampling uses ``new_size[d] = max(1, int(image_size[d] * factor))`` (see
    ``CPUImageHelper.resample``), so a downsampling factor can shrink a small
    dimension below FireANTs' floor. This returns the smallest factor that is
    ``>=`` the requested one and keeps every resampled spatial dim ``>=``
    ``min_size_voxels``.

    A dimension whose *native* size is already below ``min_size_voxels`` cannot
    be rescued by (down)sampling; for it we cap the contribution at ``1.0`` (no
    upsampling of real data past native resolution) and leave it at its native
    size — mirroring ``compute_union_bbox``, which returns the full extent when
    the image is smaller than the floor.

    Args:
        image_size: Spatial size of the image (any dim count). For a 4-D image
            pass only the spatial dims (e.g. ``size[:3]``).
        scale_factor: Requested resampling factor (fraction of native
            resolution; ``> 0``).
        min_size_voxels: Per-dimension voxel floor. ``<= 0`` disables clamping.

    Returns:
        The effective scale factor (``>=`` the requested one).
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be > 0, got {scale_factor}.")
    if min_size_voxels <= 0:
        return scale_factor

    effective = scale_factor
    for sz in image_size:
        if sz <= 0:
            continue
        if max(1, int(sz * effective)) >= min_size_voxels:
            continue
        if sz < min_size_voxels:
            # Native dim already under the floor; don't upsample real data.
            needed = 1.0
        else:
            needed = min_size_voxels / sz
            # int() floors, so a float round-down could leave us one voxel
            # short; nudge up until the resampled size truly clears the floor.
            while int(sz * needed) < min_size_voxels:
                needed = math.nextafter(needed, math.inf)
        effective = max(effective, needed)

    return effective


def pad_image_to_min_size(
    image: ImageWrapper,
    min_size_voxels: int = FIREANTS_MIN_IMG_SIZE,
    constant: float = 0.0,
) -> ImageWrapper:
    """
    Symmetrically pad any spatial dimension below ``min_size_voxels`` up to the
    floor with a constant value, so the image clears FireANTs' MIN_IMG_SIZE.

    This is the companion to ``clamp_scale_factor_for_min_size``. The clamp keeps
    *downsampling* from pushing a dimension below the floor, but it cannot rescue
    a dimension whose *native* size is already below it (you can't recover detail
    by upsampling). For those natively-thin volumes — a flat slab where, say,
    z < 32 — FireANTs would still silently and inconsistently upsample the fixed
    vs moving image and crash with a tensor shape mismatch. Padding adds empty
    border voxels so every spatial dim is at/above the floor, mirroring the
    ``min_size_voxels`` guard the roi-crop path applies to the bbox.

    Padding preserves spacing and direction; SimpleITK shifts the origin so the
    original voxels keep their physical-space location (the deformation field
    computed on the padded grid therefore still resamples correctly onto the
    high-res reference, whose extent is a subset of the padded extent).

    A no-op when every spatial dim already meets the floor (the common case), so
    well-sized studies are unaffected.

    Args:
        image: Image to pad.
        min_size_voxels: Per-dimension voxel floor. ``<= 0`` disables padding.
        constant: Fill value for the added border voxels (default background 0).

    Returns:
        The padded image (or the input unchanged if no dim was below the floor).
    """
    if min_size_voxels <= 0:
        return image

    data = image.get_data()
    if data is None:
        raise ValueError("image has no data")

    size = list(data.GetSize())
    lower = [0] * len(size)
    upper = [0] * len(size)
    needs_pad = False
    for d, sz in enumerate(size):
        if sz < min_size_voxels:
            need = min_size_voxels - sz
            lower[d] = need // 2
            upper[d] = need - lower[d]
            needs_pad = True

    if not needs_pad:
        return image

    pad = sitk.ConstantPadImageFilter()
    pad.SetPadLowerBound(lower)
    pad.SetPadUpperBound(upper)
    pad.SetConstant(constant)
    return ImageWrapper(pad.Execute(data))


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