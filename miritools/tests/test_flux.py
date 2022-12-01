import pytest
import numpy as np
from astropy import units as u
from miritools import flux as imflux
from miritools import constants

test_data = [
    (23.24912848, 13.028),
    (15.16375216, 13.492),
]


@pytest.mark.parametrize("flux, ref_mag", test_data)
def test_flux2mag(flux, ref_mag):
    """
    Test flux2mag

    Only test V-band
    """

    mag = imflux.flux2mag(flux, band="V")

    np.testing.assert_almost_equal(mag, ref_mag)


@pytest.mark.parametrize("ref_flux, mag", test_data)
def test_mag2flux(mag, ref_flux):
    """
    Test mag2flux

    Only test V-band
    """

    flux, wref = imflux.mag2flux(mag, band="V")

    np.testing.assert_almost_equal(flux, ref_flux)
    np.testing.assert_almost_equal(wref, constants.band_info["Johnson V"]["wref"])


def test_band_info():
    """
    Test band_info
    """

    band_name = "V"
    system = "Johnson"
    ref_wref = 0.55 * u.micron,
    ref_zeropoint = 3781 * u.Jy

    wref, zeropoint = imflux.get_band_info(band_name, system)

    assert u.isclose(wref, ref_wref), f"Wrong wref, expected {ref_wref}, got {wref}"
    assert u.isclose(zeropoint, ref_zeropoint), f"Wrong zeropoint, expected {ref_zeropoint}, got {zeropoint}"


def test_extrapolate_flux():
    """
    Test extrapolate flux
    """

    temperature = 5800  # K
    flux1 = 2  # mJy
    wave1 = 5  # micron
    wave2 = 10  # micron
    ref_extr_flux = 0.5703858 * u.mJy

    extr_flux = imflux.extrapolate_flux(flux1, wave1, wave2, temperature)

    assert u.isclose(extr_flux, ref_extr_flux)


test_data = [  # flux in jansky, wavelength in microns, resulting flux in photon/m2/s/microns
    (1e-3, 10., 1509.1901796421519),
    (1e-3, 15., 1006.1267864281011),
    (5e-3, 15., 5030.633932140506),
]


@pytest.mark.parametrize("in_flux, wav, ref_flux", test_data)
def test_jansky2photon(in_flux, wav, ref_flux):
    """
    Test jansky2photon
    """

    out_flux = imflux.jansky2photon(in_flux, wav)

    np.testing.assert_almost_equal(out_flux, ref_flux)


# use the same data, but inverted
@pytest.mark.parametrize("ref_flux, wav, in_flux", test_data)
def test_photon2jansky(in_flux, wav, ref_flux):
    """
    Test photon2jansky
    """

    out_flux = imflux.photon2jansky(in_flux, wav)

    np.testing.assert_almost_equal(out_flux, ref_flux)
