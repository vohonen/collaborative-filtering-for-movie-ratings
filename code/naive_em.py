"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.empty((n,K))
    
    for i in range(n):
        
        for j in range(K):
            
            diff = X[i] - mixture.mu[j]
            post[i, j] = mixture.p[j] * np.exp(np.dot(diff, diff) / (-2*mixture.var[j])) / ((2*np.pi*mixture.var[j])**(d/2))

    # compute the log-likelihood of the parameters based on data
    log_likelihood = np.log(post.sum(axis=1)).sum()
    # compute posteriors for each cluster with the parameters
    post /= post.sum(axis=1).reshape(n, 1)
    
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    n, K = post.shape
    _, d = X.shape
    post_K = post.sum(axis=0)
    
    mu = post.T @ X / post_K.reshape(K, 1)
    
    var = np.empty(K)
    for i in range(K):
        squared_error = (X - mu[i])**2
        row_squared_error = squared_error.sum(axis=1)
        var[i] = np.dot(post[:, i], row_squared_error)
    var /= (d*post_K)
    
    p = post_K / n
    
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
        mixture = mstep(X, post)
        
    return mixture, post, likelihood
