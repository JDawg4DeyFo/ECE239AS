# File: channel.py
# Author: Jacob Dennon
# Date: November 14th, 2026
# Description: This file will define channel functions. Given a complex BB input
# signal, these functions will output the complex BB RX signal.
import numpy as np
import math
import random
################################################################################
def AWGN_channel(x, var):
	"""
	Add White Gaussian Noise to the input signal

	Parameters
	----------
	x
		The input signal

	var
		Noise variance
	
	Returns
	-------
	y
		Noisy signal
	"""

	# Generate the noise
	rng = np.random.default_rng()

	sigma = np.sqrt(var) # square root of variance
	noise = rng.normal(size = len(x), scale = sigma) + 1j*rng.normal(size = len(x), scale = sigma)

	y = x + noise
	return y

def multipath_channel(x, taps, maxdelay, fc,
    *,
	return_delays=False):
	"""
	Add a wideband multipath component to the input signal

	Parameters
	----------
	x
		Input signal

	taps
		Number of paths
	
	maxdelay
		Maximum delay a tap can have. This is in samples [n]
	
	fc
		Carrier frequency in samples per second

	return_delays
		Whether or not to return delays. Default = False
	
	Returns
	-------
	y
		Multipath signal
	
	delays
		Delays of each tap. This is in samples
	"""
	# For tap = 1, just return x
	if taps == 1:
		if return_delays:
			return x, 0
		return x
	# Otherwise, we have to append some multipath components
	
	y = np.zeros(len(x) + maxdelay)
	delays = np.zeros(len(x))
	# Add multipath components
	for i in range(taps - 1):
		delay = math.floor(maxdelay * random.random())
		delays[i] = delay
		a_i = random.random()
		y[delay:-maxdelay + delay] = y[delay:-maxdelay + delay] + (a_i * np.exp(-2j * np.pi * fc * delay) * x)

	if return_delays:
		return y, delays
	return y

def full_channel(x, taps, maxdelay, fc, phase_offset, CFO):
	"""
	A channel with all major impairments
	
	Parameters
	----------
	x
		Input signal

	taps
		Number of taps in the channel

	maxdelay
		Maximum delay a tap can have. This is in samples [n]
		
	fc
		Carrier frequency in samples per second

	phase_offset
		phase offset between RX and TX
		
	CFO
		Carrier frequency offset. Includes doppler effect

	return_delays
		Whether or not to return delays. Default = False
	
	Returns
	-------
	y
		RX Signal
	"""
	fs = fc

	y = multipath_channel(x, taps, maxdelay, fc)
	n = np.arange(len(y))
	y = np.exp(1j * phase_offset) * np.exp(2j * np.pi * CFO * fs * n) * y
	y = AWGN_channel(y, 1e-7)

	return y







#################
#TO DO:

# Add default value for var in AWGN channel
# Figure out how to incorporate sampling frequency and carrier frequency correctly
# think about distribution of mutlipath gain
#	assume each h_p has same gain and uniform varying phase?
#	assume narrowband?
#	assume h_p follows a more nuanced model?