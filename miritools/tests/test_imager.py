import pytest
import numpy as np
from miritools import imager

test_data = [
    ((0, 0), (1, 1, 1032, 1024), (0, 0)),
    ((452, 50), (453, 51, 512, 512), (0, 0))  # Brightsky first pixel
]


@pytest.mark.parametrize("abs_px,sub_start,expected_result", test_data)
def test_abs_to_rel_pixels(abs_px, sub_start, expected_result):
    """
    Test abs_to_rel_pixels
    """

    header = {
        "SUBSTRT1": sub_start[0],
        "SUBSTRT2": sub_start[1],
        "SUBSIZE1": sub_start[2],
        "SUBSIZE2": sub_start[3],
    }

    rel_px = imager.abs_to_rel_pixels(abs_px, header)

    assert rel_px == expected_result


def test_abs_to_rel_pixels_fail():
    """
    Test abs_to_rel_pixels exceptions
    """
    # Correspond to a FULL array
    header = {
        "SUBSTRT1": 453,
        "SUBSTRT2": 51,
        "SUBSIZE1": 512,
        "SUBSIZE2": 512,
    }

    # First absolute pixel don't exist in brightsky and should fail
    with pytest.raises(IndexError):
        imager.abs_to_rel_pixels((451, 50), header)

    with pytest.raises(IndexError):
        imager.abs_to_rel_pixels((452, 49), header)

    with pytest.raises(IndexError):
        imager.abs_to_rel_pixels((964, 561), header)

    with pytest.raises(IndexError):
        imager.abs_to_rel_pixels((963, 562), header)


def test_crop_images():
    """
    Test crop_images
    """

    # Images with no header
    small_im = np.ones((10, 12))

    big_im = np.zeros((100, 120))
    big_im[:10, :12] = 2

    cropped_im = imager.crop_image(big_im, small_im)

    assert small_im.shape == cropped_im.shape
    assert np.all(cropped_im == 2)

    # Images with header
    small_header = {
        "SUBSTRT1": 4,
        "SUBSTRT2": 5,
    }
    big_header = {
        "SUBSTRT1": 1,
        "SUBSTRT2": 1,
    }
    small_im = np.ones((10, 12))

    big_im = np.zeros((100, 120))
    xstart = 3
    xstop = xstart + 12
    ystart = 4
    ystop = ystart + 10
    big_im[ystart:ystop, xstart:xstop] = 2

    cropped_im = imager.crop_image(big_im, small_im, big_header, small_header)

    assert small_im.shape == cropped_im.shape
    assert np.all(cropped_im == 2)

    # Ensure it crash if one header is missing
    with pytest.raises(ValueError):
        imager.crop_image(big_im, small_im, big_meta=big_header)

    with pytest.raises(ValueError):
        imager.crop_image(big_im, small_im, small_meta=small_header)


def test_find_array_intersect():
    """
    Test find_array_intersect
    """

    met_big = dict(SUBSTRT1=1, SUBSTRT2=1, SUBSIZE1=1032, SUBSIZE2=1024)
    met_medium = dict(SUBSTRT1=256, SUBSTRT2=257, SUBSIZE1=512, SUBSIZE2=512)
    met_small = dict(SUBSTRT1=512, SUBSTRT2=534, SUBSIZE1=64, SUBSIZE2=70)

    # ((xmin, xmax), (ymin, ymax))
    ref_values = ((511, 575), (533, 603))
    ((ref_xmin, ref_xmax), (ref_ymin, ref_ymax)) = ref_values

    ((xmin, xmax), (ymin, ymax)) = imager.find_array_intersect([met_big, met_medium, met_small])

    assert xmin == ref_xmin, f"xmin ({xmin}) not equal to expected value {ref_xmin}"
    assert xmax == ref_xmax, f"xmax ({xmax}) not equal to expected value {ref_xmax}"
    assert ymin == ref_ymin, f"ymin ({ymin}) not equal to expected value {ref_ymin}"
    assert ymax == ref_ymax, f"ymax ({ymax}) not equal to expected value {ref_ymax}"


def test_select_sub_image():
    """
    Test select_sub_image
    """

    # Images with no header
    big_im = np.zeros((10, 10))
    big_im[2:9, 3:10] = 2

    sub_image = imager.select_sub_image(big_im, center=(5, 6), radius=3)

    # If correctly placed, all pixels will have the same value
    assert np.all(sub_image == 2)

    # If correctly placed, no pixel outside the sub_image have a non-zero value
    assert sub_image.sum() == big_im.sum()


def test_select_sub_image_fail():
    """
    Test select_sub_image fails scenarii
    """

    # Images with no header
    big_im = np.zeros((11, 11))

    imager.select_sub_image(big_im, center=(5, 5), radius=5)

    with pytest.raises(ValueError):
        imager.select_sub_image(big_im, center=(4, 5), radius=5)

    with pytest.raises(ValueError):
        imager.select_sub_image(big_im, center=(5, 4), radius=5)

    with pytest.raises(ValueError):
        imager.select_sub_image(big_im, center=(6, 5), radius=5)

    with pytest.raises(ValueError):
        imager.select_sub_image(big_im, center=(5, 6), radius=5)


test_data = [  # (dy, dx), reference_position in pixel index (y, x)
    ((-1, 0), (1, 2)),
    ((0, 1), (2, 3))
]


@pytest.mark.parametrize("shift, ref_pos", test_data)
def test_subpixel_shift(shift, ref_pos):
    """
    Test of subpixel_shift

    All shift are done on one pixel located at (2, 2)
    Only test integer shifts to ensure X and Y are not inverted.
    """
    (dy, dx) = shift
    (yref, xref) = ref_pos

    image = np.zeros((5, 5))
    image[2, 2] = 1

    new_image = imager.subpixel_shift(image, dy, dx)

    (ymax, xmax) = np.unravel_index(np.argmax(new_image), new_image.shape)

    max_value = new_image[ymax, xmax]

    assert ymax == yref, "Y position of maximum incorrect"
    assert xmax == xref, "X position of maximum incorrect"
    assert max_value == 1, "Max value is not 1"

def test_radial_profile():
    """
    Test radial_profile
    """
    size = 25
    center = size // 2

    x_line = np.arange(size) - center
    y_line = np.arange(size) - center

    xv, yv = np.meshgrid(x_line, y_line)

    # Radial profile is simply 'R'
    image = np.sqrt(xv**2 + yv**2)

    # Test default function
    r, profile = imager.radial_profile(image, center=(center, center), bin_width=1)

    np.testing.assert_almost_equal(r, profile)

    # Test custom function
    constant = 2
    tmp_func = lambda x: constant
    r, profile = imager.radial_profile(image, center=(center, center), func=tmp_func)
    np.testing.assert_almost_equal(profile, [constant] * len(profile))

    # Test mask
    mask = np.full_like(image, False, dtype=bool)
    mask[10:12, :] = True
    image[10:12, :] = 0

    image = np.ma.MaskedArray(image, mask=mask, fill_value=np.nan)

    r, profile = imager.radial_profile(image, center=(center, center), bin_width=1)
    np.testing.assert_almost_equal(r, profile)


def test_radial_profile_center():
    """
    Test radial_profile center coordinate (x,y or y,x)

    Non regression test because a bug was found in the function when we realized that the actual centre required was
    (x,y) and not (y,x)
    """
    size = 25
    x_center = 3
    y_center = 10
    center = (y_center, x_center)

    # Radial profile is simply 'R'
    image = np.zeros((size, size))
    # non zero value only for r=1
    image[y_center+1, x_center] = 1
    image[y_center, x_center+1] = 1
    image[y_center-1, x_center] = 1
    image[y_center, x_center-1] = 1

    # Test default function
    r, profile = imager.radial_profile(image, center=center, bin_width=1)

    # If we have the correct center, profile value at r~1 (i.e second value with bin_width=1) must be non-zero due
    # to the 4 pixels set to 1 earlier
    assert profile[1] != 0., 'The center given as input in radial_profile is not the one used by the function'


def test_radial_profile_rmax():
    """
    Test radial_profile rmax if set

    Non regression test because a bug was found in the function. When rmax was set, the indices where not applied
    to the correct array (x instead of x[mask])
    """
    rmax = 5
    size = 25
    center = size // 2

    x_line = np.arange(size) - center
    y_line = np.arange(size) - center

    xv, yv = np.meshgrid(x_line, y_line)

    # Radial profile is simply 'R'
    image = np.sqrt(xv ** 2 + yv ** 2)

    # Test default function
    full_r, full_profile = imager.radial_profile(image, center=(center, center))
    r, profile = imager.radial_profile(image, center=(center, center), rmax=rmax)

    # We don't compare last bin since the truncation can impact part of its values
    n_points = len(r) - 1

    np.testing.assert_almost_equal(r[:n_points], full_r[:n_points],
                                   err_msg="Rmax parameter doesn't truncate radius array correctly")
    np.testing.assert_almost_equal(profile[:n_points], full_profile[:n_points],
                                   err_msg="Rmax parameter doesn't truncate profile array correctly")


def test_radial_profiles():
    """
    Test radial_profiles
    """
    ref_keys = set(["r", "mean", "std", "median", "variance", "max", "sum", "size"])
    x_size = 25
    y_size = 30
    x_center = x_size // 2
    y_center = y_size // 2

    x_line = np.arange(x_size) - x_center
    y_line = np.arange(y_size) - y_center

    xv, yv = np.meshgrid(x_line, y_line)

    # Radial profile is simply 'R'
    image = np.sqrt(xv**2 + yv**2)

    # Test default function
    radial_profile = imager.radial_profiles(image, center=(y_center, x_center))

    assert set(radial_profile.keys()) == ref_keys, f"Output of radial_profiles don't have expected set of keys"

    np.testing.assert_almost_equal(radial_profile["r"], radial_profile["mean"])
