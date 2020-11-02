from IPython import embed
import pytest
import numpy
from numpy import cos, sin, linspace, meshgrid, sqrt, pi, trapz

from invprob.shepplogan.shepplogan import Shepplogan2d


def test_init():
    """Take one ellipse from the Shepp-Logan phantom and test it."""

    phantom = Shepplogan2d()

    # [:6] : all r0 > 0
    # [:,7] : values of arctan are within +/- pi
    assert np.all(phantom.ellipses[:, 6] >= 0)
    assert np.all(np.abs(phantom.ellipses[:, 7]) <= np.pi)
    np.testing.assert_equal(phantom.ellipses[[0, 1, 4, 5, 6, 7, 8, 9], 8], 1)
    np.testing.assert_equal(phantom.ellipses[[0, 1, 4, 5, 6, 7, 8, 9], 9], 0)
    np.testing.assert_allclose(phantom.ellipses[[2, 3], 8], 0.95, atol=0.002)
    np.testing.assert_allclose(phantom.ellipses[2, 9], -0.31, atol=0.002)
    np.testing.assert_allclose(phantom.ellipses[3, 9], +0.31, atol=0.002)


def test_f():
    """Test values of the phantom in the feature space."""
    ph = Shepplogan2d()

    # test f(x,y) for fixed points where values are known.
    # These are mostly the centers of the ellipses and the regions of overlapping
    # ellipses.
    assert ph.f(0.7, 0) == 0  # point outside phantom
    assert ph.f(0.68, 0.0) == 2  # ellipse a
    assert ph.f(0.0, 0.0) == 1.02  # ellipse b
    assert ph.f(0.22, 0) == 1.0  # ellipse c
    assert ph.f(-0.22, 0) == 1.0  # ellipse d
    assert ph.f(0.0, 0.35) == 1.03  # ellipse e
    assert ph.f(0.0, 0.1) == 1.04  # ellipse f+e
    assert ph.f(0.0, 0.09) == 1.03  # ellipse f
    assert ph.f(0.0, -0.1) == 1.03  # ellipse g. Error in the book which says 1.04
    assert ph.f(-0.045, -0.1) == 1.01  # ellipse g+d. Error in the book which says 1.04
    assert ph.f(-0.08, -0.605) == 1.03  # ellipse h
    assert ph.f(0.0, -0.605) == 1.03  # ellipse i
    assert ph.f(0.06, -0.605) == 1.03  # ellipse j


@pytest.mark.parametrize("phi", numpy.linspace(0, 2 * numpy.pi, 11))
def test_rf(phi):
    """Test Radon transfomr of the phantom.

    The most way to cover the most details is to comapre
    anaytical form of the Radon transform with the numericall
    Radon transform (integration over feature space.).

    The smallest ellips minor/major axis is 0.023. This determines
    integration resolution.
    """
    ph = Shepplogan2d()
    np, nt = 201, 201

    # create the rotated coordinate system (p,t)
    ps = linspace(-1, 1, np)
    ts = linspace(-1, 1, nt)

    pss, tss = meshgrid(ps, ts)
    # convert (p,t) into the (x,y) of the phantom
    c, s = cos(phi), sin(phi)
    xss = c * pss - s * tss
    yss = s * pss + c * tss
    fss = ph.f(xss, yss)

    rf_nums = trapz(fss, ts, axis=0)
    rf_anas = ph.rf(ps, phi)

    numpy.testing.assert_allclose(rf_nums, rf_anas, atol=0.03)
