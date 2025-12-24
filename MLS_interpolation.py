import numpy as np
from numpy.linalg import norm, inv


def MLS_interpolation(x, xi, ui, order, c=None):
    """
    Moving Least Squares (MLS) interpolation in 2D

    Parameters
    ----------
    x : array-like, shape (2,)
        Point of interest [x, y]
    xi : array-like, shape (m, 2)
        Coordinates of m data points
    ui : array-like, shape (m,)
        Function values at data points
    order : int
        1 = linear, 2 = quadratic, 3 = cubic
    c : float or None
        Weight parameter (if None, computed automatically)

    Returns
    -------
    u : float
        Interpolated value at x
    u_derivative : ndarray, shape (2,)
        Spatial derivatives [du/dx, du/dy]
    u_ii : float
        Laplacian (d²u/dx² + d²u/dy²)
    """

    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    ui = np.asarray(ui, dtype=float).reshape(-1)

    m = len(ui)

    # number of basis functions
    n = order * 3 + (1 if order == 3 else 0)

    A = np.zeros((n, n))
    A_1 = np.zeros((n, n))
    A_2 = np.zeros((n, n))
    A_ii = np.zeros((n, n))

    B = np.zeros((n, m))
    B_1 = np.zeros((n, m))
    B_2 = np.zeros((n, m))
    B_ii = np.zeros((n, m))

    # distances
    d = np.sqrt((x[0] - xi[:, 0])**2 + (x[1] - xi[:, 1])**2)
    d_sorted = np.sort(d)

    if c is None:
        c = 2.0 * np.mean(d_sorted[:20])

    dm = c
    k = 1

    for i in range(m):
        di = d[i]

        if di > dm or di == 0:
            wi = 0.0
            w_d = 0.0
            w_dd = 0.0
        else:
            wi = (np.exp(-(di / c)**(2*k)) - np.exp(-(dm / c)**(2*k))) / \
                 (1 - np.exp(-(dm / c)**(2*k)))

            w_d = ((-2*k) * di**(2*k - 1) / c**(2*k)) * \
                  np.exp(-(di / c)**(2*k)) / \
                  (1 - np.exp(-(dm / c)**(2*k)))

            w_dd = (2*k - 1 + 2*k * di**(2*k) / c**(2*k)) * \
                   ((-2*k) * di**(2*k - 2) / c**(2*k)) * \
                   np.exp(-(di / c)**(2*k)) / \
                   (1 - np.exp(-(dm / c)**(2*k)))

        if di != 0:
            wi_1 = w_d * (x[0] - xi[i, 0]) / di
            wi_2 = w_d * (x[1] - xi[i, 1]) / di
            wi_ii = w_dd + 2 * w_d / di
        else:
            wi_1 = wi_2 = wi_ii = 0.0

        # basis functions
        if order == 1:
            p = np.array([1, xi[i, 0], xi[i, 1]])
        elif order == 2:
            p = np.array([
                1, xi[i, 0], xi[i, 1],
                xi[i, 0] * xi[i, 1],
                xi[i, 0]**2,
                xi[i, 1]**2
            ])
        elif order == 3:
            p = np.array([
                1, xi[i, 0], xi[i, 1],
                xi[i, 0] * xi[i, 1],
                xi[i, 0]**2, xi[i, 1]**2,
                xi[i, 0]**2 * xi[i, 1],
                xi[i, 1]**2 * xi[i, 0],
                xi[i, 0]**3, xi[i, 1]**3
            ])
        else:
            raise ValueError("Only LINEAR (1), QUADRATIC (2), and CUBIC (3) are supported")

        A += wi * np.outer(p, p)
        A_1 += wi_1 * np.outer(p, p)
        A_2 += wi_2 * np.outer(p, p)
        A_ii += wi_ii * np.outer(p, p)

        B[:, i] = wi * p
        B_1[:, i] = wi_1 * p
        B_2[:, i] = wi_2 * p
        B_ii[:, i] = wi_ii * p

    A_inv = inv(A)
    A_inv_1 = -A_inv @ A_1 @ A_inv
    A_inv_2 = -A_inv @ A_2 @ A_inv
    A_inv_ii = (
        2 * (A_inv @ A_1 @ A_inv @ A_1 @ A_inv +
             A_inv @ A_2 @ A_inv @ A_2 @ A_inv)
        - A_inv @ A_ii @ A_inv
    )

    # basis at x
    if order == 1:
        p = np.array([1, x[0], x[1]])
        p1 = np.array([0, 1, 0])
        p2 = np.array([0, 0, 1])
        pii = np.zeros(3)
    elif order == 2:
        p = np.array([1, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2])
        p1 = np.array([0, 1, 0, x[1], 2*x[0], 0])
        p2 = np.array([0, 0, 1, x[0], 0, 2*x[1]])
        pii = np.array([0, 0, 0, 0, 2, 2])
    else:
        p = np.array([
            1, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2,
            x[0]**2*x[1], x[1]**2*x[0], x[0]**3, x[1]**3
        ])
        p1 = np.array([0, 1, 0, x[1], 2*x[0], 0, 2*x[0]*x[1], x[1]**2, 3*x[0]**2, 0])
        p2 = np.array([0, 0, 1, x[0], 0, 2*x[1], x[0]**2, 2*x[1]*x[0], 0, 3*x[1]**2])
        pii = np.array([0, 0, 0, 0, 2, 2, 2*x[1], 2*x[0], 6*x[0], 6*x[1]])

    u = p @ A_inv @ B @ ui

    u_x = (p1 @ A_inv @ B + p @ (A_inv_1 @ B + A_inv @ B_1)) @ ui
    u_y = (p2 @ A_inv @ B + p @ (A_inv_2 @ B + A_inv @ B_2)) @ ui

    u_ii = (
        (pii @ A_inv @ B)
        + 2 * p1 @ (A_inv_1 @ B + A_inv @ B_1)
        + 2 * p2 @ (A_inv_2 @ B + A_inv @ B_2)
        + p @ (A_inv_ii @ B + A_inv @ B_ii)
        + 2 * p @ (A_inv_1 @ B_1 + A_inv_2 @ B_2)
    ) @ ui

    return u, np.array([u_x, u_y]), u_ii

