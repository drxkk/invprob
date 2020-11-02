"""Contains the Shepp-Logan phantom model in 2D and 3D.

The Shepp-Logan model is a simplr phantom model of the human brain.
It can be in 2D and 3D version.
"""
import matplotlib as mpl
import numpy as np


# plotting feacility
cmap = mpl.colors.ListedColormap(["0.9", "0.8", "0.7", "0.6", "0.5"])
bounds = [0.995, 1.005, 1.015, 1.025, 1.035, 1.045]
cmap.set_under("w")
cmap.set_over("k")
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


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
        # inthe Shepp-Logan model only ellipse c,d are rotated
        # by +/- 18 deg.
        phi0_cd = 18 * np.pi / 180

        ellipses = [
            #  x0,  y0,  a,  b,  phi0,   rho0
            [0.0, 0.0, 0.69, 0.92, 0.0, 2.00],
            [0.0, -0.0184, 0.6624, 0.874, 0.0, -0.98],
            [0.22, 0.0, 0.11, 0.31, -phi0_cd, -0.02],
            [-0.22, 0.0, 0.16, 0.41, phi0_cd, -0.02],
            [0.0, 0.35, 0.21, 0.25, 0.0, 0.01],
            [0.0, 0.1, 0.046, 0.046, 0.0, 0.01],
            [0.0, -0.1, 0.046, 0.046, 0.0, 0.01],
            [-0.08, -0.605, 0.023, 0.046, 0.0, 0.01],
            [0.0, -0.605, 0.023, 0.023, 0.0, 0.01],
            [0.06, -0.605, 0.023, 0.046, 0.0, 0.01],
        ]

        arr = np.zeros((10, 10), dtype=np.dtype("f8"), order="C")
        # Calculate handy values used to calculate f(x,y) and Radon transform:
        # [:,6] : calculate r0, a distnce of the ellipse center to origin
        # [:,7] : calculate gamma
        # [:,8] : calculate cos(phi0)
        # [:,9] : calculate sin(phi0)
        arr[:, :6] = ellipses
        arr[:, 6] = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        arr[:, 7] = np.arctan2(arr[:, 1], arr[:, 0])
        arr[:, 8] = np.cos(arr[:, 4])
        arr[:, 9] = np.sin(arr[:, 4])

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

    def rf(self, p, phi):
        out = np.zeros_like(p)

        for ellipse in self.ellipses:
            (a, b, phi0, rho0, r0, gamma) = ellipse[2:8]

            # order is important
            # suffix 'p' stands for prime
            # pp : shift-transform p coordinate
            # phip : shift-transform phi coordinate
            pp = p - r0 * np.cos(gamma - phi)
            phip = phi - phi0

            # a_sq : distance to the ellipse center after shift-transform
            # tmp[tmp<0] == 0 avoid negavit numbers iunder the sqrt(tmp)
            a_sq = (a * np.cos(phip)) ** 2 + (b * np.sin(phip)) ** 2
            tmp = a_sq - pp ** 2
            tmp[tmp < 0] = 0.0
            out += 2.0 * rho0 * a * b / a_sq * np.sqrt(tmp)

        return out
