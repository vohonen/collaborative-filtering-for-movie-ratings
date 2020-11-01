"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    
    n, _ = X.shape
    K, _ = mixture.mu.shape
    ma_X = np.ma.masked_array(X, mask=(X==0))
    # the dimension d now varies per row because there are missing rows
    row_d = ma_X.count(axis=1).reshape(n,1)
    eps = 1e-16
    

    # demean to get an (K x n x d)-array
    demeaned_X = ma_X - mixture.mu[:, None]
    
    # square and sum rows, transpose for squared error (n x K)-array
    se_X = (demeaned_X**2).sum(axis=2).T
    
    # divide with -2*variance to get the exponent of a normal variable
    exp_X = se_X / (-2*mixture.var + eps)
    
    # switch back to normal np.ndarray so the totally 0 rows get mixture.p probabilities
    exp_X = np.asarray(exp_X)
    
    # because we are dealing in the log domain the exponent is as is and the other terms are added
    # add 1e-16 for numerical stability near log(0)
    p_exp_X = exp_X + np.log(mixture.p + eps)
    
    # the normal variable denominator has a varying d per row
    denom = (row_d/2) @ np.log(2*np.pi*mixture.var + eps).reshape(1, K)
    log_p = p_exp_X - denom
    
    # get each element back to normal domain, row sum and then to log for log likelihood
    # ll = np.log(np.exp(log_p).sum(axis=1)).sum()
    ll = logsumexp(log_p, axis=1).sum()
    
    # divide all elements with the row sums in the log domain for stability and revert to normal domain
    log_post = log_p - logsumexp(log_p, axis=1).reshape(n, 1)
    post = np.exp(log_post)
    
    return post, ll
    


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    n, d = X.shape
    K, _ = mixture.mu.shape
    
    ma_X = np.ma.masked_array(X, mask=(X==0))
    C_i = np.ones((n,d)) - [X == 0] # boolean True is 1, result is a matrix with ones for each value in X
    row_d = ma_X.count(axis=1).reshape(n, 1)

    ma_post = post.T @ C_i
    mu_prop = (post.T @ X) / ma_post
    # only update means that have sufficiently many soft observations (conditional probability >= 1)
    mu = np.where(ma_post >= 1, mu_prop, mixture.mu).reshape(K, d)
    
    demeaned_X = ma_X - mu[:, None]
    se_X = np.asarray((demeaned_X**2).sum(axis=2).T)
    K_dist = (post*se_X).sum(axis=0)
    normalise = (post * row_d).sum(axis=0)
    var_prop = K_dist / normalise
    # keep a minimum variance such that some clusters don't go to 0
    var = np.where(var_prop > min_variance, var_prop, min_variance)

    p = post.sum(axis=0) / n
    
    mixture = GaussianMixture(mu, var, p)
    
    return mixture
    

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    likelihood = None
    prev_likelihood = None
    
    while (prev_likelihood is None or likelihood - prev_likelihood > (1e-6)*np.abs(likelihood)):
        
        prev_likelihood = likelihood
        
        post, likelihood = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries = 0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    post, _ = estep(X, mixture)
    weighted_fill = post @ mixture.mu
    filled_X  = np.where(X==0, weighted_fill, X)
    
    return filled_X
    
    
