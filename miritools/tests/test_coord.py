import numpy as np
import pytest

from miritools import coord


def test_filter_shift():
    """
    Test filter_shift function
    """

    input_filter = "F1000W"
    output_filter = "F770W"

    expected = (-0.14, -0.62)
    result = coord.filter_shift(input_filter, output_filter)

    np.testing.assert_array_almost_equal(result, expected)


def test_convert_filter_position():
    """
    Test convert_filter_position function
    """

    input_filter = "F1000W"
    output_filter = "F770W"
    input_coord = (0.14, 0.62)

    expected = (0.00, 0.00)
    result = coord.convert_filter_position(input_coord, input_filter, output_filter)

    np.testing.assert_array_almost_equal(result, expected)

test_data = [
    ((65, 26, 52.74), 65.447983),
    ((-65, 26, 52.74), -65.447983),
]


@pytest.mark.parametrize("input, ref", test_data)
def test_dms2dd(input, ref):
    """
    Test dms2dd function
    """

    result = coord.dms2dd(*input)

    np.testing.assert_almost_equal(result, ref, decimal=6)


test_data = [
    ((3, 12, 5), 48.0208),
    ((18, 0, 58), 270.2417),
]


@pytest.mark.parametrize("input, ref", test_data)
def test_hms2dd(input, ref):
    """
    Test hms2dd function
    """

    result = coord.hms2dd(*input)

    np.testing.assert_almost_equal(result, ref, decimal=4)
