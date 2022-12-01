import pytest
import numpy as np
from miritools import mask


def test_change_mask():
    """
    Test change_mask
    """

    input_mask = np.ones((10, 10))

    input_mask[2:4, 2:4] = 2
    input_mask[5:7, 5:7] += 2

    output_mask = mask.change_mask(input_mask, exclude_from_mask=[2])

    assert np.all(output_mask[2:4, 2:4] == 0)
    assert np.all(output_mask[5:7, 5:7] == 1)


def test_combine_masks():
    """
    Test combine_masks
    """

    m1 = np.zeros((10, 10))
    m1[2, 3] = 1
    m1[5, 8] = 1

    m2 = np.zeros_like(m1)
    m2[4, 5] = 1

    m3 = np.zeros_like(m1)
    m3[5, 8] = 1

    combined = mask.combine_masks([m1, m2, m3])

    # These 3 pixels need to be masked
    assert combined[2, 3]  # Masked in 1
    assert combined[4, 5]  # Masked in 2
    assert combined[5, 8]  # Masked in 1 and 3

    # Only 3 pixels masked
    assert combined.sum() == 3

test_data = [
    (32, 128, 256, 512),
    (1),
    (512),
    (1, 512),
]


@pytest.mark.parametrize("decomposition", test_data)
def test_decompose_mask_status(decomposition):
    """
    Test decompose_mask_status
    """

    status = np.sum(decomposition)

    result = mask.decompose_mask_status(status)

    np.testing.assert_array_equal(decomposition, result)


test_data = [
    (np.full((10, 10), 1),
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (np.full((5, 5), 1),
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (np.full((10, 10), 5),
     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
]


@pytest.mark.parametrize("input,ref", test_data)
def test_get_separated_dq_array(input, ref):
    """
    Test get_separated_dq_array
    """

    (refy, refx) = input.shape

    result = mask.get_separated_dq_array(input)

    shape = result.shape

    assert len(shape) == 3, f"Cube mask array expected to have 3 dimensions (shape: {shape})"

    (ny, nx, nbits) = shape

    assert refx == nx, f"cube mask nx = {nx} ;  not equal to expected value {refx}"
    assert refy == ny, f"cube mask ny = {ny} ;  not equal to expected value {refy}"
    assert nbits == 32, "Expected 32 slices for the cube mask"

    np.testing.assert_array_equal(result[0, 0, :], ref)


test_data = [
    (32, [5]),
    (31, [0, 1, 2, 3, 4]),
    (29, [0, 2, 3, 4])
]


@pytest.mark.parametrize("input,ref", test_data)
def test_decompose_to_bits(input, ref):
    """
    Test decompose_to_bits
    """

    result = mask.decompose_to_bits(input)

    np.testing.assert_array_equal(result, ref)


def test_extract_flag_image():
    """
    Test extract_flag_image
    """

    dq_image = np.zeros((10, 10), dtype=int)

    dq_image[2:4, 5:7] += 1
    dq_image[3:5, 5:6] += 4

    ref_1 = np.zeros_like(dq_image)
    ref_5 = np.zeros_like(dq_image)
    ref_5[3:4, 5:6] = 1
    ref_1[2:4, 5:7] = 1

    result_1 = mask.extract_flag_image(dq_image, 1)
    result_5 = mask.extract_flag_image(dq_image, 5)

    np.testing.assert_array_equal(result_1, ref_1, err_msg=f"Mask not correctly extracted with flag=1")
    np.testing.assert_array_equal(result_5, ref_5, err_msg=f"Mask not correctly extracted with flag=5")
