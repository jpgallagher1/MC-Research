# -*- coding: utf-8 -*-
"""
Module Name: test.py

Description:
    Test script for the sampler.py module.

Author: John Gallagher
Created: 2025-06-12
Last Modified: 2025-06-12
Version: 1.0.0

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sampler import Sampler, gauss_f, nongauss_f

sampler = Sampler(target_function=nongauss_f, n=100000, sigma=12, seed=1234)
sampler.metropolis(initial_val=0.1)
sampler.plot_1dim_samples(x_range=(-10, 10), n_points=10000, bins=50, which='Accepted')