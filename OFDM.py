# File: project.py
# Author: Jacob Dennon
# Date: November 12th, 2026
# Description: This file will define all the helper functions required for the
# project simulations.
import numpy as np
import math
import random
import matplotlib.pyplot as plts
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