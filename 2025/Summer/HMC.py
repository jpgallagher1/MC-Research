"""
Module Name: HMC.py

Description:
    Making a list of distributions and functions for use in numerical sampling and analysis of approximate convergence. 

Author: John Gallagher
Created: 2025-06-03
Last Modified: 
Version: 1.0.0

Dependencies:
    - numpy
    - scipy
    - matplotlib.pyplot
"""
import numpy as np
from scipy.stats import norm, uniform, expon
import matplotlib.pyplot as plt

def gaussian_distribution(mean=0, std_dev=1):
    """
    analytical gaussian distribution function

    Args:
        mean (float): Mean of the distribution.
        std_dev (float): Standard deviation of the distribution.

    Returns:
        callable: A function that computes the analytical solution of the Gaussian distribution.
    """
    return lambda x: (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def metropolis_step(x, sigma, function):
    """ Perform a single Metropolis-Hastings step.
    Args:
        x (float): Current state of the chain.
        sigma (float): Step size for the proposal distribution.
        function (callable): Target distribution function.
    
    Returns:  
        tuple: A tuple containing the new state and a boolean indicating whether the step was accepted.
    """

    proposed_x = np.random.normal(x, sigma)
    alpha = min(1, function(proposed_x)/function(x)) #why does this fraction work? Base case is we draw from the center of distributions and we want more information on the curvature.  The smaller probabilties (lower values) are associated with getting more information about the smaller regions. 
    u = np.random.uniform()
    if u < alpha:
        value = proposed_x
        accepted = True
    else:
        value = x
        accepted = False
    return value, accepted
def metropolis_sampler(initial_val, function, n=1000, sigma = 1):
    """
    Perform Metropolis-Hastings sampling.

    Args:
        initial_val (float): Initial value for the chain.
        function (callable): Target distribution function.
        n (int): Number of samples to generate.
        sigma (float): Step size for the proposal distribution.
    
    Returns:
        list: List of tuples containing sampled values and acceptance status.
    """

    results = []
    current_state = initial_val
    for i in range(0, n):
        out = metropolis_step(current_state, sigma, function)
        current_state = out[0]
        results.append(out)
    return results


class symplectic_Integrator:
    """
    A class to perform symplectic integration for Hamiltonian systems.
    Note on Hamiltonian systems:
        H(q, p) = K(p) + V(q) K(p) is the 'kintetic energy' and V(q) is the 'potential energy'.
        K(p) = 0.5* p^T M^-1 p
        V(q) = -log(f(q)) where f(q) is the target distribution function.
    Practically M = sigma^2 * I where I is the identity matrix.
    
    New propoased state is given by solving the Hamiltonian equations of motion:
        dq/dt = M^-1 p
        dp/dt = -∇ log f(q)
    """
    
    def __init__(self):
        pass
    def Euler_qstep(self, t, x, dt, sigma, function):
        """
        Perform a single Euler step for symplectic integration.

        Args:
            t (float): Current time.
            dt (float): Time step for the integration.
            x (float): Current state of the chain.
            sigma (float): Standard deviation.
            function (callable): Target distribution function.

        Returns:
            tuple: A tuple containing the new state after the Symplectic Euler step.
        """
        # Kinetic energy term
        K = 0.5 * 1/sigma**2 * np.eye(len(x))  
        # Potential energy term
        V = lambda t, x: -np.log(function(x))  
        # step for position first
        q_step = x + dt*K(t, x)
        # Step for momentum
        return x + dt*V(t, q_step)
    
    # def Euler_pstep(self, t, dt, x, sigma, function):
    #     """
    #     Perform a single Euler step for symplectic integration.

    #     Args:
    #         t (float): Current time.
    #         dt (float): Time step for the integration.
    #         x (float): Current state of the chain.
    #         sigma (float): Standard deviation.
    #         function (callable): Target distribution function.

    #     Returns:
    #         tuple: A tuple containing the new state and a boolean indicating whether the step was accepted.
    #     """
    #     # Kinetic energy term
    #     K = 0.5 * 1/sigma**2 * np.eye(len(x))  
    #     # Potential energy term
    #     V = lambda t, x: -np.log(function(x))  
    #     # step for momentum first
    #     q_step = x + dt*V(t, x)
    #     # Step for position
    #     return x + dt*K(t, q_step)
        
    def leapfrog_step(self, t,  x, dt, sigma, function):
        """
        Perform a single Leap-Frog step for symplectic integration.

        Args:
            t (float): Current time.
            dt (float): Time step for the integration.
            x (float): Current state of the chain.
            sigma (float): Standard deviation.
            function (callable): Target distribution function.

        Returns:
            tuple: A tuple containing the new state after the Symplectic Euler step.
        """
        # Kinetic energy term
        K = lambda t, x: 0.5 * 1/sigma**2 * np.eye(len(x))  
        # Potential energy term
        V = lambda t, x: -np.log(function(x))
        # Half step for position
        q_half = x + 0.5 * dt * K(t, x)
        # Full step for momentum
        p_full = x + dt * V(t, q_half)
        # Full step for position
        q_full = q_half + 0.5 * dt * K(t, p_full)
        return q_full, p_full

    def implicitMidpointFPI(t, x, dt, tol=1e-14, maxIter=20, func):
        """
        Perform an implicit midpoint step using a fixed-point iteration (FPI) method.

        Args:
            t (float): Current time.
            x (float): Current state of the chain.
            dt (float): Time step for the integration.
            tol (float): Tolerance for convergence.
            maxIter (int): Maximum number of iterations for convergence.
            sigma (float): Standard deviation.
            function (callable): Target distribution function.
        
        Returns:
            tuple: The new state after the implicit midpoint step.
        """
        yCurrent = x + dt*func(t, x)
        res = np.inf
        k = 0
        while np.linalg.norm(res)>tol and k< maxIter:
            #FPI to solve
            yNext = x + dt*func(t+0.5 * dt, 0.5*(x+yCurrent))
            res = yNext -yCurrent
            yCurrent = yNext
            k+=1
        return yCurrent

def HMC_step(y, sigma, function, symplMethod='leapfrog'):
    """
    Perform a single Hamiltonian Monte Carlo step.

    Args:
        x (float): Current state of the chain.
        sigma (float): Step size for the proposal distribution.
        function (callable): Target distribution function.

    Returns:
        tuple: A tuple containing the new state and a boolean indicating whether the step was accepted.
    """
    q_initial = y
    p_proposed = np.random.normal(np.zeros(len(y)), sigma)
    if symplMethod == 'leapfrog':
        proposed_x, p_proposed = symplectic_Integrator().leapfrog_step(0, q_initial, sigma, function)
    elif symplMethod == 'euler':
        proposed_x = symplectic_Integrator().Euler_qstep(0, q_initial, sigma, function)
    elif symplMethod == 'implicitMidpointFPI':
        proposed_x = symplectic_Integrator().implicitMidpointFPI(0, q_initial, sigma, function)
    else:
        raise ValueError("Invalid symplectic method specified.")
    alpha = min(1, function(p_proposed) / function(y))
    u = np.random.uniform()
    if u < alpha:
        value = proposed_x
        accepted = True
    else:
        value = y
        accepted = False
    return value, accepted

def HMC_sampler(initial_val, function, n=1000, sigma=1):
    """
    Perform Hamiltonian Monte Carlo sampling.

    Args:
        initial_val (float): Initial value for the chain.
        function (callable): Target distribution function.
        n (int): Number of samples to generate.
        sigma (float): Step size for the proposal distribution.

    Returns:
        list: List of tuples containing sampled values and acceptance status.
    """
    results = []
    current_state = initial_val
    for i in range(n):
        out = HMC_step(current_state, sigma, function)
        current_state = out[0]
        results.append(out)
    return results