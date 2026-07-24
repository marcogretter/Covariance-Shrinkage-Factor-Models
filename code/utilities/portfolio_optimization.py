"""
Portfolio optimization utilities.

Implements:
- Minimum variance portfolio (closed-form solution)
- Mean-variance portfolio (closed-form solution)

References:
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
"""

import numpy as np
from scipy.optimize import brentq

from utilities.covariance_utilities import (
    _validate_covariance_matrix,
)


def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum variance portfolio weights given a covariance matrix.
    In particular the weights are given by:
    w = (Σ * 1) / (1^T * Σ * 1), i.e. the solution of the optimization problem:
    min_w w^T * Σ * w, subject to 1^T * w = 1.

    Parameters:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        np.ndarray: Weights of the minimum variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )
    n = cov_matrix.shape[0]
    ones_vec = np.ones((n, 1))

    # We follow directly the formula given at the top of the function
    min_var_ptf_numerator = np.linalg.solve(cov_matrix, ones_vec)
    min_var_ptf_denominator = ones_vec.T @ min_var_ptf_numerator
    min_var_ptf_weights = min_var_ptf_numerator / min_var_ptf_denominator

    return min_var_ptf_weights.flatten()


def mean_variance_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Calculate the classic mean-variance portfolio weights given expected returns and a
    covariance matrix.

    In particular the weights solve:
    max_w mu^T * w - (gamma / 2) * w^T * Sigma * w, subject to 1^T * w = 1,
    where mu are the expected returns and gamma is the risk-aversion parameter.

    Parameters:
        expected_returns (np.ndarray): Expected returns vector.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_aversion (float): Risk-aversion parameter gamma. Must be strictly positive.

    Returns:
        np.ndarray: Weights of the mean-variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    expected_returns = np.asarray(expected_returns, dtype=float)
    if expected_returns.ndim == 2 and 1 in expected_returns.shape:
        expected_returns = expected_returns.reshape(-1)
    elif expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be one-dimensional or a single-column vector"
        )

    if expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError(
            "expected_returns and cov_matrix must refer to the same number of assets, "
            f"got {expected_returns.shape[0]} and {cov_matrix.shape[0]}"
        )

    if not np.isfinite(expected_returns).all():
        raise ValueError("expected_returns contains NaN or Inf values")

    if not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite")

    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be strictly positive, got {risk_aversion}"
        )
    
    # In order to obtain the formula at the top of the function we have to compute 
    # the following equation (as said in the file of ptf optimization seen at the lectures: first mutual fund thm)
    
    n = cov_matrix.shape[0]
    ones_vec = np.ones(n)
    A = ones_vec.T @ np.linalg.solve(cov_matrix, expected_returns)
    C = ones_vec.T @ np.linalg.solve(cov_matrix, ones_vec)
    if A >= 1e-12:
        mean_var_ptf_weights = (A / risk_aversion) * ((np.linalg.solve(cov_matrix, expected_returns)) / A) + ((1 - (A/risk_aversion)) * (np.linalg.solve(cov_matrix, ones_vec) / C))
    else:
        mean_var_ptf_weights = (np.linalg.solve(cov_matrix, expected_returns) / risk_aversion) + (np.linalg.solve(cov_matrix, ones_vec) / C)


    return mean_var_ptf_weights.flatten()

def calibration_R_A(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_gross_exposure: float = 3.0,
    lower_bound: float = 1e-4,
    upper_bound: float = 1e4
) -> float:
    """
    This function calibrates the risk aversion parameter gamma of the mean-variance portfolio
    to achieve a target gross exposure (sum of absolute weights).

    Since w* = (1/gamma) * Sigma^{-1} * mu, the gross exposure is monotonically
    decreasing in gamma, which guarantees the existence and uniqueness of the solution.

    Parameters:
        expected_returns (np.ndarray): Expected returns vector.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        target_gross_exposure (float): Desired gross exposure ||w||_1. Default is 3.0.
        lower_bound (float): Lower bound for the brentq search. Default is 1e-4.
        upper_bound (float): Upper bound for the brentq search. Default is 1e4.

    Returns:
        float: Calibrated risk aversion parameter gamma.
    
    Raises:
        ValueError: If the target gross exposure is not achievable within the given bounds.
    """

    def gross_exposure(gamma: float) -> float:
        weights = mean_variance_portfolio(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_aversion=gamma,
        )
        return float(np.abs(weights).sum())

    # Verify that the target is bracketed by the bounds before calling brentq
    ge_lower = gross_exposure(lower_bound)
    ge_upper = gross_exposure(upper_bound)

    if ge_lower < target_gross_exposure:
        raise ValueError(
            f"Target gross exposure {target_gross_exposure} is not achievable: "
            f"even at gamma={lower_bound} the gross exposure is only {ge_lower:.4f}. "
            f"Consider lowering lower_bound."
        )
    if ge_upper > target_gross_exposure:
        raise ValueError(
            f"Target gross exposure {target_gross_exposure} is not achievable: "
            f"even at gamma={upper_bound} the gross exposure is still {ge_upper:.4f}. "
            f"Consider raising upper_bound."
        )

    calibrated_gamma = brentq(
        lambda g: gross_exposure(g) - target_gross_exposure,
        a=lower_bound,
        b=upper_bound,
        xtol=1e-6,
    )

    return float(calibrated_gamma)
