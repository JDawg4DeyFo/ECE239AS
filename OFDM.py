# File: project.py
# Author: Jacob Dennon
# Date: November 12th, 2026
# Description: This file will define all the helper functions required for the
# project simulations.
import numpy as np
import math
import random
import matplotlib.pyplot as plt

################################################################################
def gen_rand_qam_symbols(N, M):
	"""
	Return N random M-QAM symbols

	Parameters
	----------
	N
		The number of symbols to return
	
	M
		The modulation order
	
	Returns
	-------
	symbols
		A set of N random M-QAM symbols
	
	constellation
		The constellation from which symbols are drawn.
	"""
	constellation = {}
	symbols = []

	# basic defs
	order = math.ceil(math.log(M, 4))
	M_clean = int(math.pow(4, order))
	M_sqrt = int(math.sqrt(M_clean))
	A = math.sqrt(3 / (2 * (M_clean - 1)))


	# defining constellation
	# Note: not gray coded
	time_to_stop = 0
	for i in range(M_sqrt):			# i determines a
		
		# check if we're over user specified M
		if (time_to_stop == 1):
			break

		for j in range(M_sqrt):		# j determines b
			idx = j + (i * M_sqrt)

			if (idx >= M):
				time_to_stop = 1
				break

			a = A * (2 * i + 1 - M_sqrt)
			b = A * (2 * j + 1 - M_sqrt)

			constellation[idx] = complex(a, b)

	# Generate N symbols
	for i in range(N):
		symbols.append(constellation[random.randint(0, M - 1)])

	return np.array(symbols), constellation
################################################################################
# OFDM Subcarriers
# Return lists of indices
# Indices are in fft shifted form
def get_subcarrier_indices(
	num_subcarriers,
	pilot_spacing,
	num_nulls_dc=1,
	num_nulls_lower=1,
	num_nulls_upper=1):
	"""
	This function is a helper function. It facilitates the construction of an
	OFDM symbol by returning appropriate indices

	Parameters
	----------

	num_subcarriers
		Number of sub carriers

	pilot_spacing
		spacing between pilots

	num_nulls_dc
		number of DC nulls

	num_nulls_lower
		number of lower band nulls

	num_nulls_upper
		number of upper band nulls
	
	"""
	indice_list = range(num_subcarriers)
	indice_set = set(indice_list) # will be used for data indices
	
	idx_subs_nulls = [0, num_subcarriers // 2, num_subcarriers // 2 + 1]
	indice_set -= set(idx_subs_nulls)

	idx_subs_pilots = indice_list[3::pilot_spacing]
	try:
		idx_subs_pilots.pop(idx_subs_pilots.index(0)) # remove DC null if there
		
	except:
		pass

	indice_set -= set(idx_subs_nulls) # use this for data set

	idx_subs_data = list(indice_set)

	return np.asarray(idx_subs_data), np.asarray(idx_subs_pilots), np.asarray(idx_subs_nulls)
################################################################################
def get_OFDM_symbol(K, pilot_spacing, sps, CP, M,
	num_nulls_dc=1,
	num_nulls_lower=1,
	num_nulls_upper=1):
	"""
		This function will return the complex baseband time domain of one OFDM 
		symbol

		Parameters
		----------
		K
			Number of sub carriers
		
		pilot_spacing
			Number of sub carriers in between pilots
	
		sps
			Samples per symbol
		
		CP
			Ratio of cyclic prefix length to ofdm symbol length

		num_nulls_dc
			Number of DC nulls

		num_nulls_lower
			Number of lower band nulls

		num_nulls_upper
			Number of upper band nulls
		"""
	# Get indices of subcarriers
	pilot_spacing = 4 # maybe should make this a parameter
	idx_subs_data, idx_subs_pilots, idx_subs_nulls = get_subcarrier_indices(
		K,
		pilot_spacing,
		num_nulls_dc=num_nulls_dc,
		num_nulls_lower=num_nulls_lower,
		num_nulls_upper=num_nulls_upper
	)

	# Generate symbols, pilots, and empty sequence
	symbols = gen_rand_qam_symbols(len(idx_subs_data), M)
	pilots = gen_rand_qam_symbols(len(idx_subs_pilots), M)
	sequence = np.zeros(K, dtype='complex')

	# Populate the sequence
	for idx, i in enumerate(idx_subs_data):
		sequence[i] = symbols[idx]
	for idx, i in enumerate(idx_subs_pilots):
		sequence[i] = pilots[idx]

	# Convert to OFDM
	x_n = np.sqrt(K) * np.fft.ifft(sequence)
	L = math.floor(K * CP)
	cyclical_prefix = x_n[-L:]
	x_n = np.concatenate((cyclical_prefix, x_n))
	
	return x_n

def get_OFDM_frame(K, Nsym, sps, CP, M, B):
	"""
	This function will return the complex baseband time domain of one general
	OFDM frame

	Parameters
	----------
	K
		Number of sub carriers
	
	Nsym
		Number of symbols
	
	sps
		Samples per symbol
	
	CP
		Ratio of cyclic prefix length to ofdm symbol length
	"""

	return 2