import pytest
import datetime as dt
import numpy as np
from miritools import utils
from miritools import constants

test_data = [
    ({'DATE-OBS': '2019-01-01', 'TIME-OBS': '17:00:00'}, dt.datetime(2019, 1, 1, 17, 0, 0)),
    ({'DATE-OBS': '2019-01-01', 'TIME-OBS': '17:00:00.7'}, dt.datetime(2019, 1, 1, 17, 0, 0))
]


@pytest.mark.parametrize("metadata,ref_time", test_data)
def test_get_exp_time(metadata, ref_time):
    """
    Test get_exp_time
    """

    # Floating point seconds are ignored and the result is just the integer part of it.
    result = utils.get_exp_time(metadata)

    assert ref_time.timestamp() == result


test_data = [
    (5.6, 1.615500580),
    (25.5, 7.356297286)
]


@pytest.mark.parametrize("wavelength, ref_size", test_data)
def test_lambda_over_d_to_pixels(wavelength, ref_size):
    """
    Test lambda_over_d_to_pixels
    """

    size = utils.lambda_over_d_to_pixels(wavelength)

    np.testing.assert_almost_equal(size, ref_size)


def test_dump_dict():
    """
    Test dump_dict
    """

    input = {
    "plot": {
        "display": True,
        "folder": "output_folder",
        "extension": "pdf",
        "save": True,
    },
    "input": {
        "folder": "input_folder",
    },
    "analysis": {
        "sub": {
            "one": 1.,
            "two": 2.,
        },
        "log_file": "results.txt",
        "version":"miricap502.__version__",
    }
    }

    ref_str = """[plot]
  display = True
  folder = output_folder
  extension = pdf
  save = True

[input]
  folder = input_folder

[analysis]
  [[sub]]
    one = 1.0
    two = 2.0

  log_file = results.txt
  version = miricap502.__version__

"""

    output = utils.dump_dict(input)

    assert output == ref_str