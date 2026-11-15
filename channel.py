# File: channel.py
# Author: Jacob Dennon
# Date: November 14th, 2026
# Description: This file will define channel functions. Given a complex BB input
# signal, these functions will output the complex BB RX signal.
import numpy as np
import math
import random
################################################################################
def AWGN_Channel(x, var):
	"""
	Add White Gaussian Noise to the input signal

	Parameters
	----------
	x
		The input signal

	var
		Noise variance
	"""

	# Generate the noise
	rng = np.random.default_rng()

	sigma = np.sqrt(var) # square root of variance
	noise = rng.normal(size = len(x), scale = sigma) + 1j*rng.normal(size = len(x), scale = sigma)

	y = x + noise
	return y