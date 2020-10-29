import numpy as np

from invprob.shepplogan.shepplogan import Shepplogan2d


def test_ellipse():
    """Take one ellipse from the Shepp-Logan phantom and test it."""

    phantom = Shepplogan2d()

    # [:6] : all r0 > 0
    # [:,7] : values of arctan are within +/- pi
    # [:, 8-11] : empirically  hard coded numbers for Shepp-Logan model. No logical
    #    numbers
    assert np.all(phantom.ellipses[:, 6] >= 0)
    assert np.all(np.abs(phantom.ellipses[:, 7]) <= np.pi)
    np.testing.assert_allclose(phantom.ellipses[:, 8], np.ones(10), atol=0.05)
    np.testing.assert_allclose(phantom.ellipses[:, 9], np.ones(10), atol=1.31)
    np.testing.assert_allclose(phantom.ellipses[:, 10], np.zeros(10), atol=0.22)
    np.testing.assert_allclose(phantom.ellipses[:, 11], np.zeros(10), atol=0.605)


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
