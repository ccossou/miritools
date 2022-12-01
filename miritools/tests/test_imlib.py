from miritools import imlib
import numpy as np


def test_2d_binned_statistic():
    """
    Test _2d_binned_statistic
    """

    nx = 20
    ny = 40

    # 1 less than nx to have an easier time testing the values because the bin center will be easy to predict
    x_sample = nx - 1
    y_sample = ny - 1

    x_range = np.arange(nx)
    y_range = np.arange(ny)

    x_values, y_values = np.meshgrid(x_range, y_range)

    x_values = x_values.flatten()
    y_values = y_values.flatten()

    values = x_values + 10 * y_values

    result = imlib._2d_binned_statistic(x_values, y_values, values, xbins=x_sample, ybins=y_sample)

    assert result['x'].ndim == 1, "x array should be 1D"
    assert result['y'].ndim == 1, "y array should be 1D"
    assert result["x"].size == nx-1, f"x sampling doesn't have the expected size (expected {nx} got {len(result['x'])})"
    assert result["y"].size == ny-1, f"y sampling doesn't have the expected size (expected {ny} got {len(result['y'])})"

    assert result['mean'].ndim == 2, "all values (except x,y) should be 2D"
    assert result["mean"].shape == (y_sample, x_sample)

    # Select pixel in the result where the mean is over only one value (i.e, everything but the last bin in both x and y
    # because the last bin has both edges
    selection = (result["size"] == 1)
    np.testing.assert_almost_equal(result["mean"][selection], values.reshape(ny, nx)[:-1, :-1][selection])
