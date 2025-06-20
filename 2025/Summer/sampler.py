"""
Module Name: sampler.py

Description:
    Making a list of distributions and functions for use in numerical sampling and analysis of approximate convergence. 

Author: John Gallagher
Created: 2025-06-03
Last Modified: 2025-06-13
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

class Sampler:
    """A class to perform Sampling using Metropolis-Hastings algorithm or Hamiltonian Monte Carlo (HMC)."""

    def __init__(self,  target_function, samples = None, n=1000, sigma=1,
                 h=0.1,  seed=1):
        """
        Initialize the Sampler.

        Args:
            target_function (callable): Target distribution function.
            samples (list): List to store sampled values.
            initial_val (float or 1-D array): Initial value for the chain.
            n (int): Number of samples to generate.
            sigma (float or 2-dim square array): Standard deviation or COV matrix for the proposal distribution. NO ERROR HANDLING YET.
            h (float): Step size for HMC.
            seed (int): Random seed for reproducibility.
        Returns:
            None
        """
        self.target_function = target_function
        self.samples = samples if samples is not None else []
        self.accepted = None
        self.rejected = None
        self.n = n
        self.sigma = sigma
        # if isinstance(self.sigma, (float, int)):
        #     self.sigma = np.array([[self.sigma]])
        # elif isinstance(self.sigma, np.ndarray):
        #     if self.sigma.ndim == 1:
        #         self.sigma = np.diag(self.sigma)
        #     elif self.sigma.ndim == 2:
        #         if not self._check_sigma_symmetric():
        #             raise ValueError("Covariance matrix must be symmetric.")
        # else:
        #     raise ValueError("Sigma must be a float, int, or a 1-D/2-D array.")
        self.h = h
        self.accepted = 0
        self.seed = seed

    def _metropolis_step(self, x):
        """ Perform a single Metropolis-Hastings step.

        Args:
            x (float): Current state of the chain.

        Returns:  
            tuple: A tuple containing the new state and a boolean indicating whether the step was accepted.
        """

        proposed_x = np.random.normal(x, self.sigma)
        alpha = min(1, self.target_function(proposed_x) / self.target_function(x))
        u = np.random.uniform()
        if u < alpha:
            return proposed_x, True
        return x, False
    def metropolis(self, initial_val):
        """
        Perform Metropolis-Hastings sampling.
        Returns:
            list: List of tuples containing sampled values and acceptance status.
        """
        np.random.seed(self.seed)
        results = []
        x = initial_val
        for _ in range(self.n):
            new_state, accepted = self._metropolis_step(x)
            results.append((new_state, accepted))
            x = new_state
        
        self.samples = pd.DataFrame(results, columns=['Value', 'Accepted'])
        self.accepted = self.samples[self.samples['Accepted']]
        self.rejected = self.samples[~self.samples['Accepted']]
        
        return self.samples
    def plot_1dim_samples(self, x_range=(-11,11), n_points=10000, bins=50,
                     which='Accepted'):
        """
        Plot the sampled values against the target distribution.

        Args:
            x_range (tuple): Range of x values for plotting the target distribution.
            n_points (int): Number of points to plot the target distribution.
            bins (int): Number of bins for the histogram of samples.

        Returns:
            None
        """
        if self.samples is None:
            raise ValueError("No samples available. Run Metropolis() first.")
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = self.target_function(x)

        fig, plt1 = plt.subplots()

        # First y-axis with histogram
        plt1.hist(self.accepted['Value'], bins=bins, density=True)
        plt1.set_ylabel('Frequency')
        plt1.set_xlabel('Sampled Values')
        plt1.set_title('Sampled Values vs Target Distribution')

        # Second y-axis with target distribution
        color = 'tab:red'
        plt2 = plt1.twinx()
        plt2.set_ylabel('Target Distribution', color=color)
        plt2.set_ylim(0, max(y) * 1.1)
        plt2.plot(x, y, label='Target Distribution', color=color)
        
        plt.show()


class Hamiltonian_system:
    """
    A class to track the hamiltonian system state and properties.
    """

    def __init__(self, position, momentum, mass=None, hamiltonian=None):
        """
        Initialize the Hamiltonian system.

        Args:
            position (np.ndarray): Initial position vector.
            momentum (np.ndarray): Initial momentum vector.
        """
        self.position = position
        self.momentum = momentum
        self.mass = mass if mass is not None else np.eye(len(momentum))
        self.hamiltonian = hamiltonian if hamiltonian is not None else self.compute_hamiltonian(self.position, self.momentum)

    def compute_hamiltonian(self, position, momentum):
        """
        Compute the Hamiltonian of the system.

        Args:
            position (np.ndarray): Position vector.
            momentum (np.ndarray): Momentum vector.

        Returns:
            float: The Hamiltonian value.
        """
        kinetic_energy = 0.5 * np.dot(momentum.T, np.linalg.solve(self.mass, momentum))
        potential_energy = -np.log(self.hamiltonian(position)) if callable(self.hamiltonian) else 0
        return kinetic_energy + potential_energy

def gauss_f(x):
    """Gaussian target distribution."""
    return np.exp(-x**2)
# Example target functions
def nongauss_f(x):
    """Non-gaussian target distribution."""
    return np.exp(-x**2 * (2 + np.sin(5*x) + np.sin(2*x)))

def gauss_2dimf(x, cov=None):
    """2D Gaussian target distribution."""
    if cov is None:
        cov = np.eye(2)
    return np.exp(-x.dot(np.linalg.inv(cov).dot(x)))