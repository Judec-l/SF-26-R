"""
MLS_interpolation.py - Refactored
Moving Least Squares interpolation for velocity fields
"""
import numpy as np
from numpy.linalg import norm, inv
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def MLS_interpolation(
        x: np.ndarray,
        xi: np.ndarray,
        ui: np.ndarray,
        order: int,
        c: Optional[float] = None
) -> Tuple[float, np.ndarray, float]:
    """
    Moving Least Squares interpolation with derivatives.

    Interpolates scattered data using polynomial basis functions
    and exponential weight function. Computes value, gradient,
    and Laplacian at query point.

    Args:
        x: Query point [x, y]
        xi: Data points, shape (m, 2)
        ui: Data values, shape (m,)
        order: Polynomial order (1=linear, 2=quadratic, 3=cubic)
        c: Length scale for weight function (auto-computed if None)

    Returns:
        Tuple of (u, grad_u, laplacian_u) where:
        - u: interpolated value
        - grad_u: gradient [du/dx, du/dy]
        - laplacian_u: Laplacian d²u/dx² + d²u/dy²
    """
    # Validate inputs
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    ui = np.asarray(ui, dtype=float).reshape(-1)

    if len(x) != 2:
        raise ValueError("Query point x must be 2D")

    if xi.shape[1] != 2:
        raise ValueError("Data points xi must have shape (m, 2)")

    if order not in [1, 2, 3]:
        raise ValueError("Order must be 1 (linear), 2 (quadratic), or 3 (cubic)")

    m = len(ui)

    # Determine number of basis functions
    if order == 1:
        n = 3
    elif order == 2:
        n = 6
    else:  # order == 3
        n = 10

    # Initialize matrices
    A = np.zeros((n, n))
    A_1 = np.zeros((n, n))  # Derivative w.r.t. x
    A_2 = np.zeros((n, n))  # Derivative w.r.t. y
    A_ii = np.zeros((n, n))  # Second derivatives

    B = np.zeros((n, m))
    B_1 = np.zeros((n, m))
    B_2 = np.zeros((n, m))
    B_ii = np.zeros((n, m))

    # Compute distances
    d = np.sqrt((x[0] - xi[:, 0]) ** 2 + (x[1] - xi[:, 1]) ** 2)
    d_sorted = np.sort(d)

    # Auto-compute length scale if not provided
    if c is None:
        c = 2.0 * np.mean(d_sorted[:min(20, len(d_sorted))])

    dm = c  # Maximum distance for weight function
    k = 1  # Exponent parameter

    # Build weighted least squares matrices
    for i in range(m):
        di = d[i]

        # Compute weight and derivatives
        if di > dm or di == 0:
            wi = 0.0
            w_d = 0.0
            w_dd = 0.0
        else:
            # Exponential weight function
            exp_term = np.exp(-(di / c) ** (2 * k))
            exp_dm = np.exp(-(dm / c) ** (2 * k))
            denominator = 1 - exp_dm

            wi = (exp_term - exp_dm) / denominator

            # First derivative of weight
            w_d = ((-2 * k) * di ** (2 * k - 1) / c ** (2 * k)) * exp_term / denominator

            # Second derivative of weight
            w_dd = ((2 * k - 1 + 2 * k * di ** (2 * k) / c ** (2 * k)) *
                    ((-2 * k) * di ** (2 * k - 2) / c ** (2 * k)) *
                    exp_term / denominator)

        # Compute directional derivatives of weight
        if di != 0:
            wi_1 = w_d * (x[0] - xi[i, 0]) / di  # dw/dx
            wi_2 = w_d * (x[1] - xi[i, 1]) / di  # dw/dy
            wi_ii = w_dd + 2 * w_d / di  # d²w/dx² + d²w/dy²
        else:
            wi_1 = 0.0
            wi_2 = 0.0
            wi_ii = 0.0

        # Compute polynomial basis at data point
        if order == 1:
            p = np.array([1, xi[i, 0], xi[i, 1]])
        elif order == 2:
            p = np.array([
                1, xi[i, 0], xi[i, 1],
                xi[i, 0] * xi[i, 1],
                xi[i, 0] ** 2,
                xi[i, 1] ** 2
            ])
        else:  # order == 3
            p = np.array([
                1, xi[i, 0], xi[i, 1],
                xi[i, 0] * xi[i, 1],
                xi[i, 0] ** 2, xi[i, 1] ** 2,
                xi[i, 0] ** 2 * xi[i, 1],
                xi[i, 1] ** 2 * xi[i, 0],
                xi[i, 0] ** 3, xi[i, 1] ** 3
            ])

        # Update matrices
        A += wi * np.outer(p, p)
        A_1 += wi_1 * np.outer(p, p)
        A_2 += wi_2 * np.outer(p, p)
        A_ii += wi_ii * np.outer(p, p)

        B[:, i] = wi * p
        B_1[:, i] = wi_1 * p
        B_2[:, i] = wi_2 * p
        B_ii[:, i] = wi_ii * p

    # Compute matrix inverse
    try:
        A_inv = inv(A)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix in MLS interpolation, using pseudoinverse")
        A_inv = np.linalg.pinv(A)

    # Compute derivatives of inverse
    A_inv_1 = -A_inv @ A_1 @ A_inv
    A_inv_2 = -A_inv @ A_2 @ A_inv
    A_inv_ii = (
            2 * (A_inv @ A_1 @ A_inv @ A_1 @ A_inv +
                 A_inv @ A_2 @ A_inv @ A_2 @ A_inv)
            - A_inv @ A_ii @ A_inv
    )

    # Compute polynomial basis and derivatives at query point
    if order == 1:
        p = np.array([1, x[0], x[1]])
        p1 = np.array([0, 1, 0])  # dp/dx
        p2 = np.array([0, 0, 1])  # dp/dy
        pii = np.zeros(3)  # d²p/dx² + d²p/dy²
    elif order == 2:
        p = np.array([1, x[0], x[1], x[0] * x[1], x[0] ** 2, x[1] ** 2])
        p1 = np.array([0, 1, 0, x[1], 2 * x[0], 0])
        p2 = np.array([0, 0, 1, x[0], 0, 2 * x[1]])
        pii = np.array([0, 0, 0, 0, 2, 2])
    else:  # order == 3
        p = np.array([
            1, x[0], x[1], x[0] * x[1], x[0] ** 2, x[1] ** 2,
                           x[0] ** 2 * x[1], x[1] ** 2 * x[0], x[0] ** 3, x[1] ** 3
        ])
        p1 = np.array([
            0, 1, 0, x[1], 2 * x[0], 0,
                           2 * x[0] * x[1], x[1] ** 2, 3 * x[0] ** 2, 0
        ])
        p2 = np.array([
            0, 0, 1, x[0], 0, 2 * x[1],
                              x[0] ** 2, 2 * x[1] * x[0], 0, 3 * x[1] ** 2
        ])
        pii = np.array([
            0, 0, 0, 0, 2, 2,
            2 * x[1], 2 * x[0], 6 * x[0], 6 * x[1]
        ])

    # Interpolate value
    u = p @ A_inv @ B @ ui

    # Interpolate gradient
    u_x = (p1 @ A_inv @ B + p @ (A_inv_1 @ B + A_inv @ B_1)) @ ui
    u_y = (p2 @ A_inv @ B + p @ (A_inv_2 @ B + A_inv @ B_2)) @ ui

    # Interpolate Laplacian
    u_ii = (
                   (pii @ A_inv @ B)
                   + 2 * p1 @ (A_inv_1 @ B + A_inv @ B_1)
                   + 2 * p2 @ (A_inv_2 @ B + A_inv @ B_2)
                   + p @ (A_inv_ii @ B + A_inv @ B_ii)
                   + 2 * p @ (A_inv_1 @ B_1 + A_inv_2 @ B_2)
           ) @ ui

    return u, np.array([u_x, u_y]), u_ii