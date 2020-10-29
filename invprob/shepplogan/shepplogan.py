"""Contains the Shepp-Logan phantom model in 2D and 3D.

The Shepp-Logan model is a simplr phantom model of the human brain.
It can be in 2D and 3D version.
"""
import numpy as np


class Shepplogan2d:
    """Shepp-Logan model in 2D.

    Shepp-Logan pahntom model is composed of 10 ellipses (named a-j). Each ellipse
    has some physical quantity value rho0. Where ellipses overlap, the value of
    rho0 is different. Chosing ellipses has a benfit, that the Radon transform
    of an ellipses has analytical form. Thanks to the linearity of Radon transform
    the Radon transform of the phantom is the sum of Rf of individual ellipses (or
    any other object), as long as the ellises rho0 is defined as a difference
    between ...

    Attributes:
    -----------
    ellipses : ndarray
        the ndarray of the ellipses

    Methodes
        f :
        Rf :
    """

    def __init__(self):
        """The Shepp-Logan model is predefined and has no parameters.

        Attributes:
        -----------
        ellipses : ndarray
            Holds the definitions of the 10 ellipses that compose Shepp-Logan
            phantom.

            Columns represent following variables:
             x0, y0, a, b, alpha, rho
            where:
                x0, y0: coordiate of the center
                a, b: ellipses axes in the x and y direction
                alpha: ellipses rotation with respect to x axis
                delta_f: the step in the value of the f function from the
                    previous ellipse
        """
        # definitions of the ellipses
        ellipses = [
            #  x0,  y0,  a,  b,  alpha,   rho
            [0.0, 0.0, 0.69, 0.92, 0.0, 2.00],
            [0.0, -0.0184, 0.6624, 0.874, 0.0, -0.98],
            [0.22, 0.0, 0.11, 0.31, -18 * np.pi / 180, -0.02],
            [-0.22, 0.0, 0.16, 0.41, 18 * np.pi / 180, -0.02],
            [0.0, 0.35, 0.21, 0.25, 0.0, 0.01],
            [0.0, 0.1, 0.046, 0.046, 0.0, 0.01],
            [0.0, -0.1, 0.046, 0.046, 0.0, 0.01],
            [-0.08, -0.605, 0.023, 0.046, 0.0, 0.01],
            [0.0, -0.605, 0.023, 0.023, 0.0, 0.01],
            [0.06, -0.605, 0.023, 0.046, 0.0, 0.01],
        ]

        arr = np.zeros((10, 12), dtype=np.dtype("f8"), order="C")
        arr[:, :6] = ellipses

        # Calculate handy values used to calculate f(x,y) and Radon transform:
        # [:,6] : calculate r0, a distnce of the ellipse center to origin
        # [:,7] : calculate phi0
        # [:,8:9] : calculate cos(alpha), sin(alpha)
        # [:, 10:11] : calculate r0*cos(phi0) and r0*sin(phi0)
        arr[:, 6] = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        arr[:, 7] = np.arctan2(arr[:, 1], arr[:, 0])
        arr[:, 8] = np.cos(arr[:, 4])
        arr[:, 9] = np.sin(arr[:, 4])
        arr[:, 10] = arr[:, 6] * np.cos(arr[:, 7])
        arr[:, 11] = arr[:, 6] * np.sin(arr[:, 7])
        self.ellipses = arr

    def f(self, x, y):
        out = np.zeros_like(x)

        for ellipse in self.ellipses:
            (x0, y0, a, b, rho0, c, s) = ellipse[[0, 1, 2, 3, 5, 8, 9]]
            xp = x - x0
            yp = y - y0

            xpp = (xp * c + yp * s) / a
            ypp = (-xp * s + yp * c) / b

            r_sq = xpp * xpp + ypp * ypp
            out += rho0 * (r_sq <= 1.0)
        return out
